import torch
import torch.nn.functional as F


def soft_erode(img):
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
    return torch.minimum(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(
        img,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
    )


def soft_open(img):
    return soft_dilate(
        soft_erode(img)
    )


def soft_skeletonize(img, iterations=10):
    img = img.clamp(0.0, 1.0)

    opened = soft_open(img)
    skeleton = F.relu(
        img - opened
    )

    for _ in range(int(iterations)):
        img = soft_erode(img)
        opened = soft_open(img)

        delta = F.relu(
            img - opened
        )

        skeleton = skeleton + F.relu(
            delta - skeleton * delta
        )

    return skeleton.clamp(0.0, 1.0)


def soft_cldice_consistency(
    student_prob,
    teacher_prob,
    iterations=10,
    downsample=2,
    eps=1e-6,
):
    if downsample > 1:
        k = int(downsample)

        student_prob = F.avg_pool2d(
            student_prob,
            kernel_size=k,
            stride=k,
        )

        teacher_prob = F.avg_pool2d(
            teacher_prob,
            kernel_size=k,
            stride=k,
        )

    student_prob = student_prob.clamp(
        0.0,
        1.0,
    )

    teacher_prob = teacher_prob.clamp(
        0.0,
        1.0,
    )

    student_skel = soft_skeletonize(
        student_prob,
        iterations=iterations,
    )

    teacher_skel = soft_skeletonize(
        teacher_prob,
        iterations=iterations,
    )

    dims = (1, 2, 3)

    topology_precision = (
        (student_skel * teacher_prob).sum(dim=dims)
        + eps
    ) / (
        student_skel.sum(dim=dims)
        + eps
    )

    topology_sensitivity = (
        (teacher_skel * student_prob).sum(dim=dims)
        + eps
    ) / (
        teacher_skel.sum(dim=dims)
        + eps
    )

    soft_cldice = (
        2.0
        * topology_precision
        * topology_sensitivity
        + eps
    ) / (
        topology_precision
        + topology_sensitivity
        + eps
    )

    score = soft_cldice.mean()
    loss = 1.0 - score

    return loss, score


def soft_truncated_distance_to_foreground(
    prob,
    iterations=8,
):
    """
    Differentiable truncated distance-like field.

    Starting from a soft foreground map, repeated max-pooling
    expands the foreground by one pixel at each iteration.

    A pixel accumulates distance while foreground has not yet reached it.
    The result is normalized to [0,1] approximately.

    This is NOT OpenCV/scipy's exact Euclidean distance transform.
    It is a differentiable truncated morphological distance field.
    """
    prob = prob.clamp(0.0, 1.0)

    reached = prob
    distance = torch.zeros_like(
        prob
    )

    for _ in range(int(iterations)):
        distance = (
            distance
            + (1.0 - reached)
        )

        reached = soft_dilate(
            reached
        )

    return (
        distance
        / float(iterations)
    ).clamp(0.0, 1.0)


def soft_signed_distance_field(
    prob,
    iterations=8,
):
    """
    Truncated signed distance-like field.

    Positive values: background side.
    Negative values: foreground side.

    Range is approximately [-1, 1].
    """
    dist_to_fg = (
        soft_truncated_distance_to_foreground(
            prob,
            iterations=iterations,
        )
    )

    dist_to_bg = (
        soft_truncated_distance_to_foreground(
            1.0 - prob,
            iterations=iterations,
        )
    )

    return (
        dist_to_fg
        - dist_to_bg
    )


def distance_field_consistency(
    student_prob,
    teacher_prob,
    iterations=8,
    downsample=2,
):
    """
    L1 consistency between differentiable truncated signed
    distance fields of Student and EMA Teacher.

    This penalizes spatial displacement and excessive crack expansion.
    """
    if downsample > 1:
        k = int(downsample)

        student_prob = F.avg_pool2d(
            student_prob,
            kernel_size=k,
            stride=k,
        )

        teacher_prob = F.avg_pool2d(
            teacher_prob,
            kernel_size=k,
            stride=k,
        )

    student_field = (
        soft_signed_distance_field(
            student_prob,
            iterations=iterations,
        )
    )

    teacher_field = (
        soft_signed_distance_field(
            teacher_prob,
            iterations=iterations,
        )
    )

    loss = F.l1_loss(
        student_field,
        teacher_field,
    )

    return loss
