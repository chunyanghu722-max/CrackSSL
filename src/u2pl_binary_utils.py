from collections import deque
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_channels, dim=64):
        super().__init__()
        hidden = max(64, int(dim))
        groups = 8
        while hidden % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, int(dim), 1, bias=True),
        )

    def forward(self, x):
        return F.normalize(self.net(x).float(), dim=1)


class CPUFeatureQueue:
    def __init__(self, max_size=10000):
        self.max_size = int(max_size)
        self.data = None

    def __len__(self):
        return 0 if self.data is None else int(self.data.shape[0])

    @torch.no_grad()
    def enqueue(self, features):
        if features is None or features.numel() == 0:
            return
        x = features.detach().float().cpu()
        if self.data is None:
            self.data = x[-self.max_size:].clone()
        else:
            self.data = torch.cat([self.data, x], dim=0)[-self.max_size:]

    def sample(self, n, device):
        if self.data is None or len(self) == 0:
            return None
        idx = torch.randint(0, len(self), (int(n),))
        return self.data[idx].to(device=device, non_blocking=True)

    def state_dict(self):
        return {"max_size": self.max_size, "data": self.data}

    def load_state_dict(self, state):
        self.max_size = int(state["max_size"])
        self.data = state["data"]


def forward_with_decoder_features(model, x):
    feats = model.encoder(x)
    try:
        dec = model.decoder(feats)
    except TypeError:
        dec = model.decoder(*feats)
    logits = model.segmentation_head(dec)
    return logits, dec


@torch.no_grad()
def infer_decoder_channels(model, height, width, device):
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 3, int(height), int(width), device=device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
        _, dec = forward_with_decoder_features(model, x)
    c = int(dec.shape[1])
    model.train(was_training)
    return c


def binary_distribution_from_logit(logits):
    p = torch.sigmoid(logits.float()).clamp(1e-6, 1 - 1e-6)
    return torch.cat([1.0 - p, p], dim=1)


def normalized_binary_entropy(prob2):
    h = -(prob2 * torch.log(prob2.clamp_min(1e-10))).sum(dim=1)
    return h / math.log(2.0)


@torch.no_grad()
def adaptive_entropy_masks(teacher_logits, epoch_index, total_epochs, initial_unreliable=0.20):
    """
    U2PL-style dynamic partition adjustment.
    Top alpha_t fraction by entropy is unreliable; the remainder is reliable.
    alpha_t linearly decreases during training.
    """
    prob2 = binary_distribution_from_logit(teacher_logits)
    entropy = normalized_binary_entropy(prob2)
    alpha_t = float(initial_unreliable) * max(0.0, 1.0 - float(epoch_index) / max(float(total_epochs), 1.0))
    flat = entropy.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.ones_like(entropy, dtype=torch.bool), torch.zeros_like(entropy, dtype=torch.bool), entropy, alpha_t
    if alpha_t <= 0.0:
        reliable = torch.ones_like(entropy, dtype=torch.bool)
        unreliable = torch.zeros_like(entropy, dtype=torch.bool)
        return reliable, unreliable, entropy, alpha_t
    q = max(0.0, min(1.0, 1.0 - alpha_t))
    gamma = torch.quantile(flat, q)
    reliable = entropy <= gamma
    unreliable = entropy > gamma
    return reliable, unreliable, entropy, alpha_t


def masked_pseudo_bce(student_logits, teacher_logits, reliable_mask):
    pseudo = (torch.sigmoid(teacher_logits.detach()) >= 0.5).float()
    per_pixel = F.binary_cross_entropy_with_logits(student_logits, pseudo, reduction="none")[:, 0]
    mask = reliable_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (per_pixel * mask).sum() / denom


def onehot_binary(mask_float, size):
    y = F.interpolate(mask_float.float(), size=size, mode="nearest")
    return torch.cat([1.0 - y, y], dim=1)


def resize_prob2(prob2, size):
    return F.interpolate(prob2.float(), size=size, mode="bilinear", align_corners=False)


def _gather_feat(z, mask):
    # z BxDxHxW, mask BxHxW
    zz = z.permute(0, 2, 3, 1)
    return zz[mask]


def u2pl_binary_contrastive(
    student_z_l, student_z_u,
    teacher_z_l, teacher_z_u,
    label_l, teacher_logits_l, teacher_logits_u,
    reliable_mask_u, unreliable_mask_u,
    queues,
    current_class_threshold=0.30,
    num_queries=32,
    num_negatives=64,
    temperature=0.50,
):
    """
    Binary-task adaptation of U2PL's category-wise memory-bank contrastive loss.

    Important adaptation:
    Original U2PL uses probability-rank interval [r_l, r_h) = [3,20) on
    multi-class datasets. With only crack/background, that interval is empty.
    Here the only valid non-top rank is used: r_l=1, r_h=2.
    Thus an unreliable pixel becomes a negative for the class ranked second.
    """
    device = student_z_l.device
    size = student_z_l.shape[-2:]

    # Teacher class distributions and hard pseudo labels at representation resolution.
    prob2_l = resize_prob2(binary_distribution_from_logit(teacher_logits_l), size)
    prob2_u = resize_prob2(binary_distribution_from_logit(teacher_logits_u), size)
    pseudo_u = prob2_u.argmax(dim=1)
    rel_u = F.interpolate(reliable_mask_u.float().unsqueeze(1), size=size, mode="nearest")[:, 0].bool()
    unrel_u = F.interpolate(unreliable_mask_u.float().unsqueeze(1), size=size, mode="nearest")[:, 0].bool()

    label_l_oh = onehot_binary(label_l, size)
    label_l_cls = label_l_oh.argmax(dim=1)

    # Teacher labeled class probabilities are not required for ground-truth membership;
    # student/teacher embeddings at labeled pixels form stable anchors/prototypes.
    losses = []

    for cls in (0, 1):
        # U2PL-style low-entropy / reliable class membership.
        # Labeled membership uses GT; unlabeled membership uses reliable pseudo labels.
        proto_l_mask = (label_l_cls == cls)
        proto_u_mask = (pseudo_u == cls) & rel_u
        anchor_l_mask = proto_l_mask & (prob2_l[:, cls] > float(current_class_threshold))
        anchor_u_mask = proto_u_mask & (prob2_u[:, cls] > float(current_class_threshold))

        t_proto_l = _gather_feat(teacher_z_l, proto_l_mask)
        t_proto_u = _gather_feat(teacher_z_u, proto_u_mask)
        s_anchor_l = _gather_feat(student_z_l, anchor_l_mask)
        s_anchor_u = _gather_feat(student_z_u, anchor_u_mask)

        proto_parts = [x for x in (t_proto_l, t_proto_u) if x.numel() > 0]
        anchor_parts = [x for x in (s_anchor_l, s_anchor_u) if x.numel() > 0]
        if not proto_parts or not anchor_parts:
            continue
        t_all = torch.cat(proto_parts, dim=0)
        s_all = torch.cat(anchor_parts, dim=0)
        prototype = F.normalize(t_all.mean(dim=0, keepdim=True), dim=1)

        # Unreliable negatives: in a 2-class problem, class cls must be rank-2.
        # rank-2 is equivalent to P(cls) < P(other class).
        other = 1 - cls
        neg_u_mask = unrel_u & (prob2_u[:, cls] < prob2_u[:, other])
        neg_u = _gather_feat(teacher_z_u, neg_u_mask)
        queues[cls].enqueue(neg_u)

        # Labeled U2PL negative: GT is not cls, while cls is teacher rank-1.
        # This is the binary equivalent of original class_mask_l using [:low_rank] with low_rank=1.
        neg_l_mask = (label_l_cls != cls) & (prob2_l[:, cls] > prob2_l[:, other])
        neg_l = _gather_feat(teacher_z_l, neg_l_mask)
        if neg_l.numel() > 0:
            # limit queue update to control background dominance
            if neg_l.shape[0] > 512:
                idx = torch.randperm(neg_l.shape[0], device=neg_l.device)[:512]
                neg_l = neg_l[idx]
            queues[cls].enqueue(neg_l)

        if len(queues[cls]) == 0:
            continue

        nq = min(int(num_queries), int(s_all.shape[0]))
        qidx = torch.randperm(s_all.shape[0], device=device)[:nq]
        anchors = F.normalize(s_all[qidx], dim=1)
        negatives = queues[cls].sample(nq * int(num_negatives), device=device)
        if negatives is None:
            continue
        negatives = F.normalize(negatives, dim=1).view(nq, int(num_negatives), -1)
        positive = prototype.unsqueeze(0).expand(nq, 1, -1)
        all_keys = torch.cat([positive, negatives], dim=1)
        logits = torch.cosine_similarity(anchors.unsqueeze(1), all_keys, dim=2) / float(temperature)
        targets = torch.zeros(nq, dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, targets))

    if not losses:
        return 0.0 * student_z_l.sum()
    return torch.stack(losses).mean()
