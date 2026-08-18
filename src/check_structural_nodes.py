import torch
from structural_topology_utils import centerline_node_consistency


def main():
    torch.manual_seed(42)
    student_logits = torch.randn(2, 1, 96, 160, requires_grad=True)
    teacher_logits = torch.randn(2, 1, 96, 160)
    student_prob = torch.sigmoid(student_logits)
    teacher_prob = torch.sigmoid(teacher_logits).detach()

    skeleton_loss, soft_cldice, node_loss, endpoint_score, junction_score, diagnostics = centerline_node_consistency(
        student_prob,
        teacher_prob,
        skeleton_iterations=5,
        downsample=1,
        node_radius=2,
    )

    total = 0.05 * skeleton_loss + 0.02 * node_loss
    total.backward()

    grad_ok = student_logits.grad is not None and torch.isfinite(student_logits.grad).all()
    maps_finite = all(torch.isfinite(v).all() for v in diagnostics.values())

    print("Structural-node topology check")
    print("skeleton loss :", float(skeleton_loss))
    print("soft clDice   :", float(soft_cldice))
    print("node loss     :", float(node_loss))
    print("endpoint score:", float(endpoint_score))
    print("junction score:", float(junction_score))
    print("maps finite   :", bool(maps_finite))
    print("gradient OK   :", bool(grad_ok))

    if not grad_ok:
        raise RuntimeError("Gradient check failed.")
    if not maps_finite:
        raise RuntimeError("Non-finite structural node maps.")
    print("STRUCTURAL NODE CHECK: OK")


if __name__ == "__main__":
    main()
