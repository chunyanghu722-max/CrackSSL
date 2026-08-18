import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjector(nn.Module):
    """
    Lightweight projection head used only during training.
    It is NOT required at inference time.
    """
    def __init__(
        self,
        in_channels,
        projection_dim=64,
    ):
        super().__init__()

        hidden = max(
            int(projection_dim),
            64,
        )

        groups = 8
        while hidden % groups != 0 and groups > 1:
            groups -= 1

        self.net = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                hidden,
            ),
            nn.ReLU(
                inplace=True,
            ),
            nn.Conv2d(
                hidden,
                int(projection_dim),
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.net(x)


def _soft_erode(img):
    p1 = -F.max_pool2d(
        -img,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
    )
    p2 = -F.max_pool2d(
        -img,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1),
    )
    return torch.minimum(
        p1,
        p2,
    )


def _soft_dilate(img):
    return F.max_pool2d(
        img,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def _soft_open(img):
    return _soft_dilate(
        _soft_erode(img)
    )


@torch.no_grad()
def soft_skeleton_for_sampling(
    probability,
    iterations=6,
):
    """
    Detached soft skeleton used only to choose contrastive samples.
    """
    img = probability.clamp(
        0.0,
        1.0,
    )

    opened = _soft_open(
        img
    )

    skeleton = F.relu(
        img - opened
    )

    for _ in range(
        int(iterations)
    ):
        img = _soft_erode(
            img
        )

        opened = _soft_open(
            img
        )

        delta = F.relu(
            img - opened
        )

        skeleton = (
            skeleton
            + F.relu(
                delta
                - skeleton * delta
            )
        )

    return skeleton.clamp(
        0.0,
        1.0,
    )


def _topk_indices(
    scores,
    candidate_mask,
    k,
):
    flat_scores = (
        scores.reshape(-1)
    )

    flat_mask = (
        candidate_mask.reshape(-1)
    )

    indices = torch.nonzero(
        flat_mask,
        as_tuple=False,
    ).squeeze(1)

    if indices.numel() == 0:
        return None

    values = flat_scores[
        indices
    ]

    count = min(
        int(k),
        int(indices.numel()),
    )

    if count <= 0:
        return None

    selected = torch.topk(
        values,
        k=count,
        largest=True,
        sorted=False,
    ).indices

    return indices[
        selected
    ]


def _gather_features(
    feature_map,
    indices,
):
    """
    feature_map: C x H x W
    return: N x C
    """
    channels = (
        feature_map.shape[0]
    )

    flat = feature_map.reshape(
        channels,
        -1,
    ).transpose(
        0,
        1,
    )

    return flat[
        indices
    ]


def _safe_cross_entropy(
    logits,
):
    labels = torch.zeros(
        logits.shape[0],
        dtype=torch.long,
        device=logits.device,
    )

    return F.cross_entropy(
        logits,
        labels,
    )


def skeleton_guided_hard_negative_contrastive(
    student_projection,
    teacher_projection,
    teacher_probability,
    temperature=0.10,
    positive_threshold=0.55,
    negative_threshold=0.30,
    foreground_threshold=0.50,
    hard_negative_radius=4,
    max_positive_samples=48,
    max_negative_samples=96,
    skeleton_iterations=6,
    global_negative_fallback=True,
):
    """
    Skeleton-guided local hard-negative InfoNCE.

    Positive crack samples:
        high-confidence Teacher pixels with strong soft-skeleton response.

    Hard negatives:
        low-probability background near the Teacher crack,
        ranked by feature similarity to the Teacher crack prototype.
        Thus the most crack-like nearby background is selected first.

    Contrastive directions:
        1) Student crack anchor:
           same-position Teacher crack = positive,
           Teacher hard backgrounds = negatives.

        2) Student hard-background anchor:
           same-position Teacher background = positive,
           Teacher crack features = negatives.

    The Teacher branch is expected to be detached / no-grad.
    """
    if student_projection.ndim != 4:
        raise ValueError(
            "student_projection must be BxCxHxW"
        )

    if teacher_projection.shape != student_projection.shape:
        raise ValueError(
            "Teacher and Student projection maps must have the same shape."
        )

    batch_size, _, height, width = (
        student_projection.shape
    )

    teacher_probability = F.interpolate(
        teacher_probability.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).clamp(
        0.0,
        1.0,
    )

    with torch.no_grad():
        teacher_skeleton = (
            soft_skeleton_for_sampling(
                teacher_probability,
                iterations=skeleton_iterations,
            )
        )

    student_projection = F.normalize(
        student_projection.float(),
        dim=1,
    )

    teacher_projection = F.normalize(
        teacher_projection.float().detach(),
        dim=1,
    )

    total_losses = []

    positive_counts = []
    negative_counts = []
    same_position_similarities = []
    hard_negative_similarities = []

    for batch_index in range(
        batch_size
    ):
        probability = (
            teacher_probability[
                batch_index,
                0,
            ]
        )

        skeleton = (
            teacher_skeleton[
                batch_index,
                0,
            ]
        )

        # Skeleton score deliberately multiplies confidence:
        # centerline points with stronger Teacher support rank first.
        positive_score = (
            skeleton
            * probability
        )

        positive_candidate = (
            probability
            >= float(
                positive_threshold
            )
        ) & (
            positive_score > 1e-6
        )

        positive_indices = (
            _topk_indices(
                positive_score,
                positive_candidate,
                max_positive_samples,
            )
        )

        if positive_indices is None:
            continue

        foreground = (
            probability
            >= float(
                foreground_threshold
            )
        ).float().view(
            1,
            1,
            height,
            width,
        )

        radius = int(
            hard_negative_radius
        )

        kernel = (
            2 * radius + 1
        )

        nearby = F.max_pool2d(
            foreground,
            kernel_size=kernel,
            stride=1,
            padding=radius,
        )[0, 0] > 0.0

        negative_candidate = (
            nearby
            & (
                probability
                <= float(
                    negative_threshold
                )
            )
        )

        student_feature = (
            student_projection[
                batch_index
            ]
        )

        teacher_feature = (
            teacher_projection[
                batch_index
            ]
        )

        teacher_positive = (
            _gather_features(
                teacher_feature,
                positive_indices,
            )
        )

        crack_prototype = F.normalize(
            teacher_positive.mean(
                dim=0,
                keepdim=True,
            ),
            dim=1,
        )

        negative_indices_all = torch.nonzero(
            negative_candidate.reshape(-1),
            as_tuple=False,
        ).squeeze(1)

        if (
            negative_indices_all.numel() == 0
            and bool(
                global_negative_fallback
            )
        ):
            global_negative = (
                probability
                <= float(
                    negative_threshold
                )
            )

            negative_indices_all = torch.nonzero(
                global_negative.reshape(-1),
                as_tuple=False,
            ).squeeze(1)

        if negative_indices_all.numel() == 0:
            continue

        teacher_negative_all = (
            _gather_features(
                teacher_feature,
                negative_indices_all,
            )
        )

        # Actual hard-negative mining:
        # choose background features MOST SIMILAR to crack prototype.
        hard_scores = (
            teacher_negative_all
            @ crack_prototype.transpose(
                0,
                1,
            )
        ).squeeze(1)

        negative_count = min(
            int(max_negative_samples),
            int(
                negative_indices_all.numel()
            ),
        )

        hardest_order = torch.topk(
            hard_scores,
            k=negative_count,
            largest=True,
            sorted=False,
        ).indices

        negative_indices = (
            negative_indices_all[
                hardest_order
            ]
        )

        teacher_negative = (
            teacher_negative_all[
                hardest_order
            ]
        )

        student_positive = (
            _gather_features(
                student_feature,
                positive_indices,
            )
        )

        student_negative = (
            _gather_features(
                student_feature,
                negative_indices,
            )
        )

        # ----------------------------------------------------
        # Direction A: crack anchors vs hard background keys
        # ----------------------------------------------------
        positive_logit = (
            student_positive
            * teacher_positive
        ).sum(
            dim=1,
            keepdim=True,
        )

        negative_logits = (
            student_positive
            @ teacher_negative.transpose(
                0,
                1,
            )
        )

        crack_logits = torch.cat(
            [
                positive_logit,
                negative_logits,
            ],
            dim=1,
        ) / float(
            temperature
        )

        crack_loss = (
            _safe_cross_entropy(
                crack_logits
            )
        )

        # ----------------------------------------------------
        # Direction B: hard-bg anchors vs crack keys
        # ----------------------------------------------------
        background_positive_logit = (
            student_negative
            * teacher_negative
        ).sum(
            dim=1,
            keepdim=True,
        )

        background_negative_logits = (
            student_negative
            @ teacher_positive.transpose(
                0,
                1,
            )
        )

        background_logits = torch.cat(
            [
                background_positive_logit,
                background_negative_logits,
            ],
            dim=1,
        ) / float(
            temperature
        )

        background_loss = (
            _safe_cross_entropy(
                background_logits
            )
        )

        image_loss = (
            0.5 * crack_loss
            + 0.5 * background_loss
        )

        total_losses.append(
            image_loss
        )

        positive_counts.append(
            float(
                positive_indices.numel()
            )
        )

        negative_counts.append(
            float(
                negative_indices.numel()
            )
        )

        same_position_similarities.append(
            float(
                positive_logit.detach()
                .mean()
                .item()
            )
        )

        hard_negative_similarities.append(
            float(
                hard_scores[
                    hardest_order
                ].detach()
                .mean()
                .item()
            )
        )

    if not total_losses:
        zero = (
            student_projection.sum()
            * 0.0
        )

        stats = {
            "valid_images": 0.0,
            "positive_samples": 0.0,
            "negative_samples": 0.0,
            "positive_similarity": 0.0,
            "hard_negative_similarity": 0.0,
        }

        return zero, stats

    loss = torch.stack(
        total_losses
    ).mean()

    def mean_or_zero(values):
        if not values:
            return 0.0
        return float(
            sum(values)
            / len(values)
        )

    stats = {
        "valid_images": float(
            len(total_losses)
        ),
        "positive_samples": (
            mean_or_zero(
                positive_counts
            )
        ),
        "negative_samples": (
            mean_or_zero(
                negative_counts
            )
        ),
        "positive_similarity": (
            mean_or_zero(
                same_position_similarities
            )
        ),
        "hard_negative_similarity": (
            mean_or_zero(
                hard_negative_similarities
            )
        ),
    }

    return loss, stats
