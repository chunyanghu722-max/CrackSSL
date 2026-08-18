import torch
import torch.nn.functional as F


def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iterations=10):
    """Differentiable soft skeletonization. Input/output: Bx1xHxW in [0,1]."""
    img = img.clamp(0.0, 1.0)
    opened = soft_open(img)
    skeleton = F.relu(img - opened)
    for _ in range(int(iterations)):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def soft_cldice_from_precomputed_skeletons(student_prob, teacher_prob, student_skel, teacher_skel, eps=1e-6):
    dims = (1, 2, 3)
    topology_precision = ((student_skel * teacher_prob).sum(dim=dims) + eps) / (student_skel.sum(dim=dims) + eps)
    topology_sensitivity = ((teacher_skel * student_prob).sum(dim=dims) + eps) / (teacher_skel.sum(dim=dims) + eps)
    soft_cldice = (2.0 * topology_precision * topology_sensitivity + eps) / (topology_precision + topology_sensitivity + eps)
    score = soft_cldice.mean()
    return 1.0 - score, score


def soft_neighbor_degree(skeleton):
    """Soft 8-neighborhood degree: endpoint~1, line~2, junction>=3."""
    kernel = torch.ones((1, 1, 3, 3), dtype=skeleton.dtype, device=skeleton.device)
    neighborhood_sum = F.conv2d(skeleton, kernel, padding=1)
    return (neighborhood_sum - skeleton).clamp(min=0.0, max=8.0)


def structural_node_maps(
    skeleton,
    endpoint_alpha=4.0,
    junction_beta=8.0,
    junction_threshold=2.5,
    suppression_floor=0.05,
):
    """Differentiable Endpoint and Junction response maps from a soft skeleton."""
    skeleton = skeleton.clamp(0.0, 1.0)
    degree = soft_neighbor_degree(skeleton)

    endpoint_raw = skeleton * torch.exp(-float(endpoint_alpha) * (degree - 1.0).pow(2))
    junction_raw = skeleton * torch.sigmoid(float(junction_beta) * (degree - float(junction_threshold)))

    floor = float(suppression_floor)
    endpoint = (F.relu(endpoint_raw - floor) / max(1.0 - floor, 1e-6)).clamp(0.0, 1.0)
    junction = (F.relu(junction_raw - floor) / max(1.0 - floor, 1e-6)).clamp(0.0, 1.0)
    return endpoint, junction, degree


def tolerant_node_f1(student_nodes, teacher_nodes, radius=2, eps=1e-6):
    """Spatially tolerant structural-node F1. Allows small 1-2 pixel node shifts."""
    radius = int(radius)
    if radius > 0:
        k = 2 * radius + 1
        teacher_region = F.max_pool2d(teacher_nodes, kernel_size=k, stride=1, padding=radius)
        student_region = F.max_pool2d(student_nodes, kernel_size=k, stride=1, padding=radius)
    else:
        teacher_region = teacher_nodes
        student_region = student_nodes

    dims = (1, 2, 3)
    student_mass = student_nodes.sum(dim=dims)
    teacher_mass = teacher_nodes.sum(dim=dims)

    precision = ((student_nodes * teacher_region).sum(dim=dims) + eps) / (student_mass + eps)
    recall = ((teacher_nodes * student_region).sum(dim=dims) + eps) / (teacher_mass + eps)
    score = (2.0 * precision * recall + eps) / (precision + recall + eps)

    both_empty = (student_mass < 1e-5) & (teacher_mass < 1e-5)
    score = torch.where(both_empty, torch.ones_like(score), score)
    return score.mean()


def centerline_node_consistency(
    student_prob,
    teacher_prob,
    skeleton_iterations=10,
    downsample=2,
    node_radius=2,
    endpoint_alpha=4.0,
    junction_beta=8.0,
    junction_threshold=2.5,
):
    """Joint centerline + endpoint/junction structural consistency."""
    if downsample > 1:
        k = int(downsample)
        student_prob = F.avg_pool2d(student_prob, kernel_size=k, stride=k)
        teacher_prob = F.avg_pool2d(teacher_prob, kernel_size=k, stride=k)

    student_prob = student_prob.clamp(0.0, 1.0)
    teacher_prob = teacher_prob.clamp(0.0, 1.0)

    student_skel = soft_skeletonize(student_prob, iterations=skeleton_iterations)
    teacher_skel = soft_skeletonize(teacher_prob, iterations=skeleton_iterations)

    skeleton_loss, soft_cldice_score = soft_cldice_from_precomputed_skeletons(
        student_prob, teacher_prob, student_skel, teacher_skel
    )

    student_endpoint, student_junction, student_degree = structural_node_maps(
        student_skel,
        endpoint_alpha=endpoint_alpha,
        junction_beta=junction_beta,
        junction_threshold=junction_threshold,
    )
    teacher_endpoint, teacher_junction, teacher_degree = structural_node_maps(
        teacher_skel,
        endpoint_alpha=endpoint_alpha,
        junction_beta=junction_beta,
        junction_threshold=junction_threshold,
    )

    endpoint_score = tolerant_node_f1(student_endpoint, teacher_endpoint, radius=node_radius)
    junction_score = tolerant_node_f1(student_junction, teacher_junction, radius=node_radius)
    node_score = 0.5 * endpoint_score + 0.5 * junction_score
    node_loss = 1.0 - node_score

    diagnostics = {
        "student_skeleton": student_skel,
        "teacher_skeleton": teacher_skel,
        "student_endpoint": student_endpoint,
        "teacher_endpoint": teacher_endpoint,
        "student_junction": student_junction,
        "teacher_junction": teacher_junction,
        "student_degree": student_degree,
        "teacher_degree": teacher_degree,
    }

    return skeleton_loss, soft_cldice_score, node_loss, endpoint_score, junction_score, diagnostics
