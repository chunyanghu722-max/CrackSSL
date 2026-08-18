import torch

from contrastive_utils import (
    ContrastiveProjector,
    skeleton_guided_hard_negative_contrastive,
)


def main():
    torch.manual_seed(42)

    batch = 2
    channels = 64
    height = 48
    width = 80

    student_feature = torch.randn(
        batch,
        channels,
        height,
        width,
        requires_grad=True,
    )

    teacher_feature = torch.randn(
        batch,
        channels,
        height,
        width,
    )

    student_projector = ContrastiveProjector(
        channels,
        projection_dim=32,
    )

    teacher_projector = ContrastiveProjector(
        channels,
        projection_dim=32,
    )

    teacher_projector.load_state_dict(
        student_projector.state_dict()
    )

    for parameter in (
        teacher_projector.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    teacher_prob = torch.full(
        (
            batch,
            1,
            height,
            width,
        ),
        0.05,
    )

    # Synthetic thin cracks.
    teacher_prob[
        0,
        0,
        23:26,
        10:68,
    ] = 0.92

    teacher_prob[
        1,
        0,
        8:42,
        38:41,
    ] = 0.90

    # Add low-probability local background ring.
    teacher_prob[
        0,
        0,
        19:30,
        7:71,
    ] = torch.maximum(
        teacher_prob[
            0,
            0,
            19:30,
            7:71,
        ],
        torch.tensor(0.18),
    )

    teacher_prob[
        0,
        0,
        23:26,
        10:68,
    ] = 0.92

    teacher_prob[
        1,
        0,
        5:45,
        34:45,
    ] = torch.maximum(
        teacher_prob[
            1,
            0,
            5:45,
            34:45,
        ],
        torch.tensor(0.16),
    )

    teacher_prob[
        1,
        0,
        8:42,
        38:41,
    ] = 0.90

    student_projection = (
        student_projector(
            student_feature
        )
    )

    with torch.no_grad():
        teacher_projection = (
            teacher_projector(
                teacher_feature
            )
        )

    loss, stats = (
        skeleton_guided_hard_negative_contrastive(
            student_projection,
            teacher_projection,
            teacher_prob,
            temperature=0.10,
            positive_threshold=0.55,
            negative_threshold=0.30,
            hard_negative_radius=4,
            max_positive_samples=48,
            max_negative_samples=96,
        )
    )

    loss.backward()

    grad_ok = (
        student_feature.grad
        is not None
        and torch.isfinite(
            student_feature.grad
        ).all()
    )

    projector_grad_ok = any(
        parameter.grad is not None
        and torch.isfinite(
            parameter.grad
        ).all()
        for parameter in (
            student_projector.parameters()
        )
    )

    print(
        "Skeleton-guided contrastive check"
    )
    print(
        "loss                    :",
        float(
            loss.detach()
        ),
    )
    print(
        "valid images            :",
        stats[
            "valid_images"
        ],
    )
    print(
        "positive samples/image  :",
        stats[
            "positive_samples"
        ],
    )
    print(
        "negative samples/image  :",
        stats[
            "negative_samples"
        ],
    )
    print(
        "positive similarity     :",
        stats[
            "positive_similarity"
        ],
    )
    print(
        "hard-negative similarity:",
        stats[
            "hard_negative_similarity"
        ],
    )
    print(
        "feature gradient OK     :",
        bool(
            grad_ok
        ),
    )
    print(
        "projector gradient OK   :",
        bool(
            projector_grad_ok
        ),
    )

    if not grad_ok:
        raise RuntimeError(
            "Feature gradient check failed."
        )

    if not projector_grad_ok:
        raise RuntimeError(
            "Projection-head gradient check failed."
        )

    if stats["valid_images"] <= 0:
        raise RuntimeError(
            "No valid synthetic contrastive samples."
        )

    print(
        "CONTRASTIVE CHECK: OK"
    )


if __name__ == "__main__":
    main()
