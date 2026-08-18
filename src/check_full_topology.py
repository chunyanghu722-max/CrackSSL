import torch

from topology_utils import (
    distance_field_consistency,
    soft_cldice_consistency,
)


def main():
    torch.manual_seed(42)

    student_logits = torch.randn(
        2, 1, 96, 160,
        requires_grad=True,
    )

    teacher_logits = torch.randn(
        2, 1, 96, 160,
    )

    student_prob = torch.sigmoid(
        student_logits
    )

    teacher_prob = torch.sigmoid(
        teacher_logits
    ).detach()

    skel_loss, skel_score = (
        soft_cldice_consistency(
            student_prob,
            teacher_prob,
            iterations=5,
            downsample=1,
        )
    )

    dist_loss = (
        distance_field_consistency(
            student_prob,
            teacher_prob,
            iterations=6,
            downsample=1,
        )
    )

    total = (
        skel_loss
        + 0.05 * dist_loss
    )

    total.backward()

    grad_ok = (
        student_logits.grad is not None
        and torch.isfinite(
            student_logits.grad
        ).all()
    )

    print("Full topology utility check")
    print(
        "skeleton loss :",
        float(skel_loss),
    )
    print(
        "soft clDice   :",
        float(skel_score),
    )
    print(
        "distance loss :",
        float(dist_loss),
    )
    print(
        "gradient OK   :",
        bool(grad_ok),
    )

    if not grad_ok:
        raise RuntimeError(
            "Gradient check failed."
        )

    print(
        "FULL TOPOLOGY CHECK: OK"
    )


if __name__ == "__main__":
    main()
