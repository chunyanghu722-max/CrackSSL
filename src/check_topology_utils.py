import torch

from topology_utils import (
    soft_skeletonize,
    soft_cldice_consistency,
)


def main():
    torch.manual_seed(42)

    student_logits = torch.randn(
        2,
        1,
        96,
        160,
        requires_grad=True,
    )

    teacher_logits = torch.randn(
        2,
        1,
        96,
        160,
    )

    student_prob = torch.sigmoid(
        student_logits
    )

    teacher_prob = torch.sigmoid(
        teacher_logits
    ).detach()

    loss, score = (
        soft_cldice_consistency(
            student_prob,
            teacher_prob,
            iterations=5,
            downsample=1,
        )
    )

    loss.backward()

    grad_ok = (
        student_logits.grad
        is not None
        and torch.isfinite(
            student_logits.grad
        ).all()
    )

    print(
        "Topology utility test"
    )
    print(
        "loss       :",
        float(loss),
    )
    print(
        "soft clDice:",
        float(score),
    )
    print(
        "gradient OK:",
        bool(grad_ok),
    )

    if not grad_ok:
        raise RuntimeError(
            "Topology gradient check failed."
        )

    print("TOPOLOGY CHECK: OK")


if __name__ == "__main__":
    main()
