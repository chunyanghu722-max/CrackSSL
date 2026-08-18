import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize


def _neighbors8(y, x, h, w):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w:
                yield yy, xx


def _find_endpoints(skeleton):
    """
    Return endpoint coordinates (y, x) on a binary one-pixel skeleton.
    Endpoint = skeleton pixel with exactly one 8-neighbor.
    """
    sk = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(
        sk,
        ddepth=cv2.CV_16S,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    ) - sk.astype(np.int16)

    ys, xs = np.where(
        (sk > 0) & (neighbor_count == 1)
    )
    return list(zip(ys.tolist(), xs.tolist()))


def _trace_endpoint_direction(skeleton, endpoint, max_steps=6):
    """
    Estimate the outward tangent direction of an endpoint.

    We walk inward along the skeleton for several pixels, then use:
        outward = endpoint - interior_point
    """
    h, w = skeleton.shape
    start = endpoint

    previous = None
    current = start
    path = [start]

    for _ in range(int(max_steps)):
        candidates = []
        for yy, xx in _neighbors8(
            current[0], current[1], h, w
        ):
            if not skeleton[yy, xx]:
                continue
            if previous is not None and (yy, xx) == previous:
                continue
            candidates.append((yy, xx))

        if not candidates:
            break

        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            # Prefer the point furthest from the original endpoint.
            ey, ex = start
            nxt = max(
                candidates,
                key=lambda p: (
                    (p[0] - ey) ** 2
                    + (p[1] - ex) ** 2
                ),
            )

        previous = current
        current = nxt
        path.append(current)

    if len(path) < 2:
        return None

    interior = path[-1]

    # Convert from (y, x) to Cartesian-like (x, y).
    vx = float(start[1] - interior[1])
    vy = float(start[0] - interior[0])

    norm = math.sqrt(vx * vx + vy * vy)
    if norm < 1e-6:
        return None

    return np.array(
        [vx / norm, vy / norm],
        dtype=np.float32,
    )


def _line_pixels(p1, p2, h, w):
    """
    Return integer pixels along line p1 -> p2.
    p1/p2 are (y, x).
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.line(
        mask,
        (int(p1[1]), int(p1[0])),
        (int(p2[1]), int(p2[0])),
        color=1,
        thickness=1,
        lineType=cv2.LINE_8,
    )

    ys, xs = np.where(mask > 0)
    return ys, xs, mask.astype(bool)


def _pair_score(
    p1,
    p2,
    d1,
    d2,
    teacher_prob,
    min_distance=2.0,
    max_distance=12.0,
    min_alignment=0.45,
    min_line_prob=0.08,
):
    """
    Score whether two endpoints form a plausible short crack gap.

    Conditions:
    - sufficiently close;
    - each endpoint's outward tangent points roughly toward the other;
    - Teacher retains weak probability support along the bridge.
    """
    y1, x1 = p1
    y2, x2 = p2

    vx = float(x2 - x1)
    vy = float(y2 - y1)

    distance = math.sqrt(vx * vx + vy * vy)

    if (
        distance < float(min_distance)
        or distance > float(max_distance)
    ):
        return None

    direction12 = np.array(
        [vx / distance, vy / distance],
        dtype=np.float32,
    )
    direction21 = -direction12

    align1 = float(
        np.dot(d1, direction12)
    )
    align2 = float(
        np.dot(d2, direction21)
    )

    if (
        align1 < float(min_alignment)
        or align2 < float(min_alignment)
    ):
        return None

    h, w = teacher_prob.shape
    ys, xs, _ = _line_pixels(
        p1, p2, h, w
    )

    if len(xs) == 0:
        return None

    line_prob = float(
        teacher_prob[ys, xs].mean()
    )

    if line_prob < float(min_line_prob):
        return None

    mean_alignment = 0.5 * (
        align1 + align2
    )

    # Higher is better.
    distance_score = (
        1.0
        - distance / float(max_distance)
    )

    score = (
        0.45 * mean_alignment
        + 0.35 * line_prob
        + 0.20 * distance_score
    )

    return {
        "score": score,
        "distance": distance,
        "alignment": mean_alignment,
        "line_prob": line_prob,
    }


def build_gap_targets_single(
    teacher_prob,
    threshold=0.5,
    min_component_area=4,
    max_distance=12,
    min_alignment=0.45,
    min_line_prob=0.08,
    tangent_steps=6,
    bridge_thickness=1,
    guard_radius=2,
    guard_bg_threshold=0.20,
    max_pairs=4,
):
    """
    Generate conservative crack-gap repair targets from one Teacher map.

    Returns:
        bridge_mask: positive target (fill a short gap)
        guard_mask : negative target around bridge, only where Teacher says background
        metadata
    """
    teacher_prob = np.asarray(
        teacher_prob,
        dtype=np.float32,
    )

    binary = (
        teacher_prob >= float(threshold)
    )

    # Remove tiny foreground components before topology analysis.
    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary.astype(np.uint8),
            connectivity=8,
        )
    )

    clean = np.zeros_like(
        binary,
        dtype=bool,
    )

    component_id_map = np.zeros_like(
        labels,
        dtype=np.int32,
    )

    new_id = 1

    for label_id in range(
        1, n_labels
    ):
        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < int(min_component_area):
            continue

        region = (
            labels == label_id
        )

        clean[region] = True
        component_id_map[region] = (
            new_id
        )
        new_id += 1

    skeleton = skeletonize(
        clean
    ).astype(bool)

    endpoints = _find_endpoints(
        skeleton
    )

    endpoint_info = []

    for endpoint in endpoints:
        direction = (
            _trace_endpoint_direction(
                skeleton,
                endpoint,
                max_steps=tangent_steps,
            )
        )

        if direction is None:
            continue

        y, x = endpoint
        component_id = int(
            component_id_map[y, x]
        )

        if component_id <= 0:
            # Skeletonization can theoretically shift topology at corners.
            # Search a tiny local neighborhood for component identity.
            found = 0
            for yy, xx in _neighbors8(
                y, x,
                clean.shape[0],
                clean.shape[1],
            ):
                cid = int(
                    component_id_map[
                        yy, xx
                    ]
                )
                if cid > 0:
                    found = cid
                    break
            component_id = found

        if component_id <= 0:
            continue

        endpoint_info.append({
            "point": endpoint,
            "direction": direction,
            "component_id": component_id,
        })

    candidates = []

    for i in range(
        len(endpoint_info)
    ):
        for j in range(
            i + 1,
            len(endpoint_info),
        ):
            a = endpoint_info[i]
            b = endpoint_info[j]

            # Only bridge different connected Teacher skeleton components.
            if (
                a["component_id"]
                == b["component_id"]
            ):
                continue

            pair = _pair_score(
                a["point"],
                b["point"],
                a["direction"],
                b["direction"],
                teacher_prob,
                max_distance=max_distance,
                min_alignment=min_alignment,
                min_line_prob=min_line_prob,
            )

            if pair is None:
                continue

            candidates.append({
                "i": i,
                "j": j,
                **pair,
            })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    used_endpoints = set()
    selected = []

    for candidate in candidates:
        if len(selected) >= int(max_pairs):
            break

        i = candidate["i"]
        j = candidate["j"]

        if (
            i in used_endpoints
            or j in used_endpoints
        ):
            continue

        selected.append(
            candidate
        )

        used_endpoints.add(i)
        used_endpoints.add(j)

    h, w = teacher_prob.shape

    bridge_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    pair_metadata = []

    for candidate in selected:
        p1 = endpoint_info[
            candidate["i"]
        ]["point"]

        p2 = endpoint_info[
            candidate["j"]
        ]["point"]

        cv2.line(
            bridge_mask,
            (
                int(p1[1]),
                int(p1[0]),
            ),
            (
                int(p2[1]),
                int(p2[0]),
            ),
            color=1,
            thickness=int(
                bridge_thickness
            ),
            lineType=cv2.LINE_8,
        )

        pair_metadata.append({
            "p1": [
                int(p1[0]),
                int(p1[1]),
            ],
            "p2": [
                int(p2[0]),
                int(p2[1]),
            ],
            "distance": float(
                candidate["distance"]
            ),
            "alignment": float(
                candidate["alignment"]
            ),
            "line_prob": float(
                candidate["line_prob"]
            ),
            "score": float(
                candidate["score"]
            ),
        })

    # Only supervise pixels that are actually missing in Teacher's hard foreground.
    bridge_mask = (
        (bridge_mask > 0)
        & (~clean)
    )

    # Guard band: a narrow ring around the bridge.
    # It is used as background target only if Teacher probability is low,
    # preventing a repaired gap from becoming a thick blob.
    if bridge_mask.any():
        kernel_size = (
            2 * int(guard_radius)
            + 1
        )

        kernel = np.ones(
            (
                kernel_size,
                kernel_size,
            ),
            dtype=np.uint8,
        )

        dilated = cv2.dilate(
            bridge_mask.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)

        guard_mask = (
            dilated
            & (~bridge_mask)
            & (
                teacher_prob
                < float(
                    guard_bg_threshold
                )
            )
            & (~clean)
        )
    else:
        guard_mask = np.zeros_like(
            bridge_mask,
            dtype=bool,
        )

    metadata = {
        "num_endpoints": len(
            endpoint_info
        ),
        "num_candidate_pairs": len(
            candidates
        ),
        "num_selected_pairs": len(
            selected
        ),
        "bridge_pixels": int(
            bridge_mask.sum()
        ),
        "guard_pixels": int(
            guard_mask.sum()
        ),
        "pairs": pair_metadata,
    }

    return (
        bridge_mask.astype(
            np.float32
        ),
        guard_mask.astype(
            np.float32
        ),
        metadata,
    )


def gap_repair_loss(
    student_prob,
    teacher_prob,
    threshold=0.5,
    downsample=2,
    max_distance=12,
    min_alignment=0.45,
    min_line_prob=0.08,
    guard_weight=0.5,
):
    """
    Online conservative gap-repair loss.

    Teacher topology proposal generation is discrete and detached.
    Student probability remains differentiable.

    Positive bridge:
        encourage Student probability -> 1.

    Negative guard band:
        encourage Student probability -> 0.

    Returns:
        total gap loss
        positive bridge loss
        guard loss
        average selected gap count
        average bridge pixel ratio
    """
    if downsample > 1:
        k = int(downsample)

        student_small = F.avg_pool2d(
            student_prob,
            kernel_size=k,
            stride=k,
        )

        teacher_small = F.avg_pool2d(
            teacher_prob,
            kernel_size=k,
            stride=k,
        )
    else:
        student_small = student_prob
        teacher_small = teacher_prob

    teacher_np = (
        teacher_small.detach()
        .float()
        .cpu()
        .numpy()
    )

    bridge_masks = []
    guard_masks = []
    gap_counts = []

    batch_size = (
        teacher_np.shape[0]
    )

    for index in range(
        batch_size
    ):
        bridge, guard, meta = (
            build_gap_targets_single(
                teacher_np[
                    index, 0
                ],
                threshold=threshold,
                max_distance=max_distance,
                min_alignment=min_alignment,
                min_line_prob=min_line_prob,
            )
        )

        bridge_masks.append(
            bridge
        )
        guard_masks.append(
            guard
        )
        gap_counts.append(
            meta[
                "num_selected_pairs"
            ]
        )

    bridge_tensor = torch.from_numpy(
        np.stack(
            bridge_masks,
            axis=0,
        )
    ).unsqueeze(1).to(
        student_small.device,
        dtype=student_small.dtype,
    )

    guard_tensor = torch.from_numpy(
        np.stack(
            guard_masks,
            axis=0,
        )
    ).unsqueeze(1).to(
        student_small.device,
        dtype=student_small.dtype,
    )

    eps = 1e-6

    bridge_mass = (
        bridge_tensor.sum()
    )

    guard_mass = (
        guard_tensor.sum()
    )

    if bridge_mass.item() > 0:
        bridge_loss = -(
            bridge_tensor
            * torch.log(
                student_small.clamp(
                    min=eps,
                    max=1.0 - eps,
                )
            )
        ).sum() / (
            bridge_mass + eps
        )
    else:
        bridge_loss = (
            student_small.sum()
            * 0.0
        )

    if guard_mass.item() > 0:
        guard_loss = -(
            guard_tensor
            * torch.log(
                (
                    1.0
                    - student_small
                ).clamp(
                    min=eps,
                    max=1.0 - eps,
                )
            )
        ).sum() / (
            guard_mass + eps
        )
    else:
        guard_loss = (
            student_small.sum()
            * 0.0
        )

    total_loss = (
        bridge_loss
        + float(guard_weight)
        * guard_loss
    )

    mean_gap_count = float(
        np.mean(
            gap_counts
        )
    )

    bridge_ratio = float(
        bridge_mass.item()
        / max(
            bridge_tensor.numel(),
            1,
        )
    )

    return (
        total_loss,
        bridge_loss,
        guard_loss,
        mean_gap_count,
        bridge_ratio,
    )
