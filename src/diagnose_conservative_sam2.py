from pathlib import Path
from contextlib import nullcontext
import argparse
import csv

import cv2
import numpy as np
import torch
from tqdm import tqdm

from baseline_utils import build_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def letterbox_params(h, w, target_h, target_w):
    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pad_h = target_h - new_h
    pad_w = target_w - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return {
        "new_h": new_h,
        "new_w": new_w,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def teacher_preprocess(image_rgb, target_h, target_w):
    h, w = image_rgb.shape[:2]
    p = letterbox_params(h, w, target_h, target_w)

    resized = cv2.resize(
        image_rgb,
        (p["new_w"], p["new_h"]),
        interpolation=cv2.INTER_LINEAR,
    )

    padded = cv2.copyMakeBorder(
        resized,
        p["top"],
        p["bottom"],
        p["left"],
        p["right"],
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    x = padded.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(
        x.transpose(2, 0, 1)
    ).float().unsqueeze(0)

    return tensor, p


def restore_probability(prob_letterbox, params, orig_h, orig_w):
    top = params["top"]
    left = params["left"]
    new_h = params["new_h"]
    new_w = params["new_w"]

    cropped = prob_letterbox[
        top : top + new_h,
        left : left + new_w,
    ]

    return cv2.resize(
        cropped.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )


def iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)

    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0

    return float(
        np.logical_and(a, b).sum()
        / union
    )


def component_positive_points(component, n_points=3):
    """
    Select points distributed along the dominant axis of the crack component.
    This is more informative for elongated cracks than a single center click.
    """
    ys, xs = np.where(component)

    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    xy = np.column_stack([xs, ys]).astype(np.float32)

    if len(xy) <= n_points:
        return xy

    centered = xy - xy.mean(axis=0, keepdims=True)

    cov = np.cov(centered, rowvar=False)

    if np.ndim(cov) < 2:
        return xy[[len(xy) // 2]]

    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]

    projection = centered @ axis
    order = np.argsort(projection)

    quantiles = np.linspace(
        0.15,
        0.85,
        n_points,
    )

    chosen = []
    for q in quantiles:
        index = order[
            int(
                round(
                    q * (len(order) - 1)
                )
            )
        ]
        chosen.append(xy[index])

    return np.asarray(
        chosen,
        dtype=np.float32,
    )


def component_negative_points(component, box, max_points=4):
    """
    Negative points are sampled near box corners / side centers,
    but only outside a dilated teacher component.
    """
    h, w = component.shape

    dilated = cv2.dilate(
        component.astype(np.uint8),
        np.ones((7, 7), np.uint8),
        iterations=1,
    ).astype(bool)

    x1, y1, x2, y2 = (
        box.astype(int).tolist()
    )

    candidates = [
        (x1 + 2, y1 + 2),
        (x2 - 2, y1 + 2),
        (x1 + 2, y2 - 2),
        (x2 - 2, y2 - 2),
        ((x1 + x2) // 2, y1 + 2),
        ((x1 + x2) // 2, y2 - 2),
        (x1 + 2, (y1 + y2) // 2),
        (x2 - 2, (y1 + y2) // 2),
    ]

    result = []

    for x, y in candidates:
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        if not dilated[y, x]:
            result.append(
                [float(x), float(y)]
            )

        if len(result) >= max_points:
            break

    return np.asarray(
        result,
        dtype=np.float32,
    )


def extract_components(
    binary_mask,
    min_area=8,
    box_pad=6,
    max_components=8,
):
    mask_u8 = binary_mask.astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    h, w = binary_mask.shape
    components = []

    for label_id in range(1, n_labels):
        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT,
            ]
        )
        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP,
            ]
        )
        bw = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH,
            ]
        )
        bh = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        component = (
            labels == label_id
        )

        x1 = max(
            0,
            x - box_pad,
        )
        y1 = max(
            0,
            y - box_pad,
        )
        x2 = min(
            w - 1,
            x + bw - 1 + box_pad,
        )
        y2 = min(
            h - 1,
            y + bh - 1 + box_pad,
        )

        box = np.array(
            [x1, y1, x2, y2],
            dtype=np.float32,
        )

        positive_points = (
            component_positive_points(
                component,
                n_points=3,
            )
        )

        negative_points = (
            component_negative_points(
                component,
                box,
                max_points=4,
            )
        )

        components.append({
            "area": area,
            "component": component,
            "box": box,
            "positive_points": positive_points,
            "negative_points": negative_points,
        })

    components.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    if max_components > 0:
        components = (
            components[
                :max_components
            ]
        )

    return components


def trust_region(component, radius=10):
    kernel_size = (
        2 * int(radius) + 1
    )

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8,
    )

    return cv2.dilate(
        component.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)


def select_safe_candidate(
    masks,
    scores,
    teacher_component,
    box,
    trust_radius=10,
    min_coverage=0.55,
    min_area_ratio=0.45,
    max_area_ratio=1.80,
):
    """
    Choose the SAM2 candidate that:
      1. stays near the teacher component,
      2. still covers most teacher crack pixels,
      3. does not explode in area.

    If no candidate passes the gates, return None.
    """
    region = trust_region(
        teacher_component,
        radius=trust_radius,
    )

    x1, y1, x2, y2 = (
        box.astype(int).tolist()
    )

    box_region = np.zeros_like(
        teacher_component,
        dtype=bool,
    )

    box_region[
        y1 : y2 + 1,
        x1 : x2 + 1,
    ] = True

    allowed = np.logical_and(
        region,
        box_region,
    )

    teacher_area = max(
        int(teacher_component.sum()),
        1,
    )

    best = None

    for idx in range(len(masks)):
        candidate = (
            masks[idx].astype(bool)
        )

        # Critical safety constraint:
        # SAM2 may only modify pixels close to Teacher's crack.
        candidate = np.logical_and(
            candidate,
            allowed,
        )

        candidate_area = int(
            candidate.sum()
        )

        if candidate_area == 0:
            continue

        intersection = int(
            np.logical_and(
                candidate,
                teacher_component,
            ).sum()
        )

        coverage = (
            intersection
            / teacher_area
        )

        area_ratio = (
            candidate_area
            / teacher_area
        )

        overlap = iou(
            candidate,
            teacher_component,
        )

        sam_score = float(
            scores[idx]
        )

        accepted = (
            coverage >= min_coverage
            and area_ratio >= min_area_ratio
            and area_ratio <= max_area_ratio
        )

        if not accepted:
            continue

        # Prefer candidates close to Teacher spatially,
        # while still using SAM2's own quality estimate.
        composite = (
            0.45 * overlap
            + 0.35 * coverage
            + 0.20 * sam_score
        )

        item = {
            "mask": candidate,
            "sam_score": sam_score,
            "coverage": coverage,
            "area_ratio": area_ratio,
            "overlap": overlap,
            "composite": composite,
        }

        if (
            best is None
            or composite > best["composite"]
        ):
            best = item

    return best


def make_visualization(
    image,
    teacher_mask,
    refined_mask,
    components,
    accepted_flags,
):
    teacher_vis = (
        teacher_mask.astype(np.uint8)
        * 255
    )

    refined_vis = (
        refined_mask.astype(np.uint8)
        * 255
    )

    teacher_rgb = np.repeat(
        teacher_vis[..., None],
        3,
        axis=2,
    )

    refined_rgb = np.repeat(
        refined_vis[..., None],
        3,
        axis=2,
    )

    prompt_vis = image.copy()

    for item, accepted in zip(
        components,
        accepted_flags,
    ):
        x1, y1, x2, y2 = (
            item["box"]
            .astype(int)
            .tolist()
        )

        # green box = SAM2 refinement accepted
        # red box   = rejected, Teacher kept
        box_color = (
            (0, 255, 0)
            if accepted
            else (255, 0, 0)
        )

        cv2.rectangle(
            prompt_vis,
            (x1, y1),
            (x2, y2),
            box_color,
            2,
        )

        for x, y in item[
            "positive_points"
        ].astype(int):
            cv2.circle(
                prompt_vis,
                (x, y),
                3,
                (0, 255, 0),
                -1,
            )

        for x, y in item[
            "negative_points"
        ].astype(int):
            cv2.circle(
                prompt_vis,
                (x, y),
                3,
                (255, 0, 0),
                -1,
            )

    overlay = image.copy()

    crack = refined_mask.astype(bool)

    overlay[crack] = (
        0.45 * overlay[crack]
        + 0.55
        * np.array(
            [255, 0, 0],
            dtype=np.float32,
        )
    ).astype(np.uint8)

    return np.concatenate(
        [
            image,
            teacher_rgb,
            prompt_vis,
            refined_rgb,
            overlay,
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher-checkpoint",
        default="results/ema_seed42_10pct/checkpoints/best.pt",
    )
    parser.add_argument(
        "--unlabeled-list",
        default="splits/Crack500/seed42/unlabeled_10pct.txt",
    )
    parser.add_argument(
        "--data-root",
        default="datasets/Crack500",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="sam2_repo/checkpoints/sam2.1_hiera_small.pt",
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument(
        "--teacher-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--box-pad",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--trust-radius",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output",
        default="results/sam2_conservative_diagnostic_10pct",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.teacher_checkpoint,
        map_location=device,
        weights_only=False,
    )

    cfg = checkpoint["config"]

    target_h = int(
        cfg["height"]
    )
    target_w = int(
        cfg["width"]
    )

    teacher = build_model(
        pretrained=False,
    ).to(device)

    teacher.load_state_dict(
        checkpoint["teacher_model"]
    )

    teacher.eval()

    print("=" * 94)
    print("CONSERVATIVE EMA TEACHER -> SAM2.1 REFINEMENT")
    print("=" * 94)
    print("Teacher checkpoint :", args.teacher_checkpoint)
    print("Teacher threshold  :", args.teacher_threshold)
    print("Box pad            :", args.box_pad)
    print("Trust radius       :", args.trust_radius)
    print("Positive points    : up to 3/component")
    print("Negative points    : up to 4/component")
    print("Diagnostic images  :", args.num_images)
    print("=" * 94)

    sam2_model = build_sam2(
        args.sam2_config,
        args.sam2_checkpoint,
        device=str(device),
    )

    predictor = SAM2ImagePredictor(
        sam2_model
    )

    stems = [
        line.strip()
        for line in Path(
            args.unlabeled_list
        )
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    n = min(
        args.num_images,
        len(stems),
    )

    indices = np.linspace(
        0,
        len(stems) - 1,
        n,
    ).round().astype(int)

    selected = [
        stems[i]
        for i in indices
    ]

    out_dir = Path(
        args.output
    )

    vis_dir = (
        out_dir
        / "visualizations"
    )

    vis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for stem in tqdm(
        selected,
        desc="Conservative SAM2",
    ):
        image_path = (
            Path(args.data_root)
            / "traincrop"
            / f"{stem}.jpg"
        )

        bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            raise RuntimeError(
                f"Cannot read {image_path}"
            )

        image = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        h, w = image.shape[:2]

        tensor, params = (
            teacher_preprocess(
                image,
                target_h,
                target_w,
            )
        )

        tensor = tensor.to(
            device,
            non_blocking=True,
        )

        with torch.inference_mode():
            amp = (
                torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                )
                if device.type == "cuda"
                else nullcontext()
            )

            with amp:
                logits = teacher(
                    tensor
                )

            prob_lb = (
                torch.sigmoid(
                    logits.float()
                )[0, 0]
                .cpu()
                .numpy()
            )

        teacher_prob = (
            restore_probability(
                prob_lb,
                params,
                h,
                w,
            )
        )

        teacher_mask = (
            teacher_prob
            >= args.teacher_threshold
        )

        components = (
            extract_components(
                teacher_mask,
                min_area=args.min_component_area,
                box_pad=args.box_pad,
                max_components=args.max_components,
            )
        )

        refined = (
            teacher_mask.copy()
        )

        accepted_flags = []
        accepted_count = 0
        rejected_count = 0
        component_rows = []

        if components:
            with torch.inference_mode():
                amp2 = (
                    torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                    )
                    if device.type == "cuda"
                    else nullcontext()
                )

                with amp2:
                    predictor.set_image(
                        image
                    )

                    for comp_idx, item in enumerate(
                        components
                    ):
                        pos = item[
                            "positive_points"
                        ]
                        neg = item[
                            "negative_points"
                        ]

                        if len(neg) > 0:
                            points = np.concatenate(
                                [pos, neg],
                                axis=0,
                            )
                            labels = np.concatenate(
                                [
                                    np.ones(
                                        len(pos),
                                        dtype=np.int32,
                                    ),
                                    np.zeros(
                                        len(neg),
                                        dtype=np.int32,
                                    ),
                                ]
                            )
                        else:
                            points = pos
                            labels = np.ones(
                                len(pos),
                                dtype=np.int32,
                            )

                        masks, scores, _ = (
                            predictor.predict(
                                point_coords=points,
                                point_labels=labels,
                                box=item["box"],
                                multimask_output=True,
                            )
                        )

                        chosen = (
                            select_safe_candidate(
                                masks,
                                scores,
                                item["component"],
                                item["box"],
                                trust_radius=args.trust_radius,
                            )
                        )

                        if chosen is None:
                            accepted_flags.append(
                                False
                            )
                            rejected_count += 1

                            component_rows.append({
                                "component": comp_idx,
                                "accepted": 0,
                                "teacher_area": item["area"],
                                "sam_score": "",
                                "coverage": "",
                                "area_ratio": "",
                                "overlap": "",
                            })

                            # original Teacher component remains
                            continue

                        accepted_flags.append(
                            True
                        )
                        accepted_count += 1

                        # Replace only this component.
                        refined[
                            item["component"]
                        ] = False

                        refined |= chosen[
                            "mask"
                        ]

                        component_rows.append({
                            "component": comp_idx,
                            "accepted": 1,
                            "teacher_area": item["area"],
                            "sam_score": chosen["sam_score"],
                            "coverage": chosen["coverage"],
                            "area_ratio": chosen["area_ratio"],
                            "overlap": chosen["overlap"],
                        })
        else:
            accepted_flags = []

        teacher_ratio = float(
            teacher_mask.mean()
        )
        refined_ratio = float(
            refined.mean()
        )

        global_iou = iou(
            teacher_mask,
            refined,
        )

        canvas = (
            make_visualization(
                image,
                teacher_mask,
                refined,
                components,
                accepted_flags,
            )
        )

        out_path = (
            vis_dir
            / f"{stem}_conservative_sam2.jpg"
        )

        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(
                canvas,
                cv2.COLOR_RGB2BGR,
            ),
        )

        rows.append({
            "stem": stem,
            "teacher_fg_ratio": teacher_ratio,
            "refined_fg_ratio": refined_ratio,
            "teacher_refined_iou": global_iou,
            "components": len(components),
            "accepted_components": accepted_count,
            "rejected_components": rejected_count,
        })

    stats_path = (
        out_dir
        / "diagnostic_stats.csv"
    )

    with stats_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 94)
    print("CONSERVATIVE DIAGNOSTIC COMPLETE")
    print("=" * 94)
    print("Visualizations:", vis_dir)
    print("Statistics    :", stats_path)
    print("\nPanel order:")
    print(
        "Original | Teacher mask | constrained prompts | "
        "Conservative SAM2 mask | overlay"
    )
    print("\nPrompt colors:")
    print("Green points = positive")
    print("Red points   = negative")
    print("Green box    = SAM2 candidate accepted")
    print("Red box      = SAM2 rejected; Teacher mask retained")
    print("=" * 94)


if __name__ == "__main__":
    main()
