import numpy as np
import torch

from gap_topology_utils import (
    build_gap_targets_single,
    gap_repair_loss,
)


def synthetic_teacher():
    """
    Create two short horizontal crack segments with a small gap.
    Probability inside the missing bridge is weak but nonzero.
    """
    prob = np.zeros(
        (64, 96),
        dtype=np.float32,
    )

    # left segment
    prob[32, 20:42] = 0.95

    # right segment
    prob[32, 48:72] = 0.95

    # latent weak Teacher evidence inside the gap
    prob[32, 42:48] = 0.18

    # slight smoothing / thickness
    prob[31:34, 20:42] = np.maximum(
        prob[31:34, 20:42],
        0.75,
    )
    prob[31:34, 48:72] = np.maximum(
        prob[31:34, 48:72],
        0.75,
    )

    return prob


def main():
    teacher_np = synthetic_teacher()

    bridge, guard, meta = (
        build_gap_targets_single(
            teacher_np,
            threshold=0.5,
            max_distance=12,
            min_alignment=0.45,
            min_line_prob=0.08,
        )
    )

    print(
        "Synthetic gap proposal check"
    )
    print(
        "selected pairs :",
        meta[
            "num_selected_pairs"
        ],
    )
    print(
        "bridge pixels  :",
        int(
            bridge.sum()
        ),
    )
    print(
        "guard pixels   :",
        int(
            guard.sum()
        ),
    )

    if (
        meta[
            "num_selected_pairs"
        ] < 1
    ):
        raise RuntimeError(
            "Failed to detect synthetic crack gap."
        )

    teacher = torch.from_numpy(
        teacher_np
    ).view(
        1, 1, 64, 96
    )

    student_logits = torch.full(
        (1, 1, 64, 96),
        -2.0,
        requires_grad=True,
    )

    # Existing crack segments are already reasonably predicted.
    with torch.no_grad():
        student_logits[
            0, 0, 31:34, 20:42
        ] = 2.0
        student_logits[
            0, 0, 31:34, 48:72
        ] = 2.0

    student_prob = torch.sigmoid(
        student_logits
    )

    (
        total,
        bridge_loss,
        guard_loss,
        mean_gaps,
        bridge_ratio,
    ) = gap_repair_loss(
        student_prob,
        teacher,
        threshold=0.5,
        downsample=1,
        max_distance=12,
        min_alignment=0.45,
        min_line_prob=0.08,
        guard_weight=0.5,
    )

    total.backward()

    grad_ok = (
        student_logits.grad
        is not None
        and torch.isfinite(
            student_logits.grad
        ).all()
    )

    print(
        "total loss     :",
        float(total),
    )
    print(
        "bridge loss    :",
        float(bridge_loss),
    )
    print(
        "guard loss     :",
        float(guard_loss),
    )
    print(
        "mean gaps      :",
        mean_gaps,
    )
    print(
        "bridge ratio   :",
        bridge_ratio,
    )
    print(
        "gradient OK    :",
        bool(grad_ok),
    )

    if not grad_ok:
        raise RuntimeError(
            "Gap repair gradient check failed."
        )

    print(
        "GAP REPAIR CHECK: OK"
    )


if __name__ == "__main__":
    main()
