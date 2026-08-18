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
        "scale": scale,
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
    x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0)

    return x, p


def restore_probability_to_original(prob_letterbox, params, orig_h, orig_w):
    top = params["top"]
    left = params["left"]
    new_h = params["new_h"]
    new_w = params["new_w"]

    cropped = prob_letterbox[
        top : top + new_h,
        left : left + new_w,
    ]

    restored = cv2.resize(
        cropped.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )

    return np.clip(restored, 0.0, 1.0)


def component_prompts(binary_mask, min_area=8, box_pad=10, max_components=8):
    """
    Returns the largest connected crack components.
    Each prompt contains:
      - component mask
      - padded XYXY box
      - one guaranteed foreground point from distance transform
    """
    mask_u8 = binary_mask.astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    h, w = binary_mask.shape
    components = []

    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        comp = labels == label_id

        x1 = max(0, x - box_pad)
        y1 = max(0, y - box_pad)
        x2 = min(w - 1, x + bw - 1 + box_pad)
        y2 = min(h - 1, y + bh - 1 + box_pad)

        dist = cv2.distanceTransform(
            comp.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        px, py = max_loc

        components.append({
            "area": area,
            "component": comp,
            "box": np.array(
                [x1, y1, x2, y2],
                dtype=np.float32,
            ),
            "point": np.array(
                [[float(px), float(py)]],
                dtype=np.float32,
            ),
        })

    components.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    if max_components > 0:
        components = components[:max_components]

    return components


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)

    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0

    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def make_visualization(
    image,
    teacher_mask,
    refined_mask,
    components,
):
    teacher_vis = (
        teacher_mask.astype(np.uint8) * 255
    )
    refined_vis = (
        refined_mask.astype(np.uint8) * 255
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

    for item in components:
        x1, y1, x2, y2 = (
            item["box"].astype(int).tolist()
        )
        px, py = (
            item["point"][0].astype(int).tolist()
        )

        cv2.rectangle(
            prompt_vis,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )
        cv2.circle(
            prompt_vis,
            (px, py),
            4,
            (0, 255, 0),
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
        default="results/sam2_refine_diagnostic_10pct",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    teacher_ckpt = torch.load(
        args.teacher_checkpoint,
        map_location=device,
        weights_only=False,
    )

    cfg = teacher_ckpt["config"]
    target_h = int(cfg["height"])
    target_w = int(cfg["width"])

    teacher = build_model(
        pretrained=False,
    ).to(device)

    teacher.load_state_dict(
        teacher_ckpt["teacher_model"]
    )

    teacher.eval()

    print("=" * 88)
    print("EMA TEACHER -> AUTO PROMPTS -> SAM2.1 REFINEMENT DIAGNOSTIC")
    print("=" * 88)
    print("Teacher checkpoint :", args.teacher_checkpoint)
    print("Unlabeled list     :", args.unlabeled_list)
    print("Teacher input      :", f"{target_h}x{target_w}")
    print("Teacher threshold  :", args.teacher_threshold)
    print("SAM2 checkpoint    :", args.sam2_checkpoint)
    print("Images to inspect  :", args.num_images)
    print("Device             :", device)
    print("=" * 88)

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
        for line in Path(args.unlabeled_list)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    if not stems:
        raise RuntimeError(
            "Unlabeled list is empty."
        )

    n = min(
        args.num_images,
        len(stems),
    )

    # Evenly spaced deterministic diagnostic images:
    # avoids hand-picking favorable examples.
    indices = np.linspace(
        0,
        len(stems) - 1,
        n,
    ).round().astype(int)

    selected_stems = [
        stems[i]
        for i in indices
    ]

    output_dir = Path(args.output)
    image_output_dir = (
        output_dir
        / "visualizations"
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    amp_context = (
        torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        )
        if device.type == "cuda"
        else nullcontext()
    )

    for stem in tqdm(
        selected_stems,
        desc="Refine",
    ):
        image_path = (
            Path(args.data_root)
            / "traincrop"
            / f"{stem}.jpg"
        )

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Cannot read: {image_path}"
            )

        image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        orig_h, orig_w = (
            image.shape[:2]
        )

        teacher_input, letterbox_info = (
            teacher_preprocess(
                image,
                target_h,
                target_w,
            )
        )

        teacher_input = teacher_input.to(
            device,
            non_blocking=True,
        )

        with torch.inference_mode():
            with (
                torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                )
                if device.type == "cuda"
                else nullcontext()
            ):
                teacher_logits = teacher(
                    teacher_input
                )

            teacher_prob_lb = (
                torch.sigmoid(
                    teacher_logits.float()
                )[0, 0]
                .cpu()
                .numpy()
            )

        teacher_prob = (
            restore_probability_to_original(
                teacher_prob_lb,
                letterbox_info,
                orig_h,
                orig_w,
            )
        )

        teacher_mask = (
            teacher_prob
            >= args.teacher_threshold
        )

        components = component_prompts(
            teacher_mask,
            min_area=args.min_component_area,
            box_pad=args.box_pad,
            max_components=args.max_components,
        )

        # Start from teacher mask.
        # Each selected component will be replaced by SAM2's refinement.
        refined_mask = teacher_mask.copy()

        sam_scores = []

        if components:
            with torch.inference_mode(), amp_context:
                predictor.set_image(
                    image
                )

                for item in components:
                    # Remove this coarse component before inserting SAM2 result.
                    refined_mask[
                        item["component"]
                    ] = False

                    masks, scores, _ = (
                        predictor.predict(
                            point_coords=item["point"],
                            point_labels=np.array(
                                [1],
                                dtype=np.int32,
                            ),
                            box=item["box"],
                            multimask_output=True,
                        )
                    )

                    best_index = int(
                        np.argmax(scores)
                    )

                    sam_mask = masks[
                        best_index
                    ].astype(bool)

                    sam_scores.append(
                        float(
                            scores[best_index]
                        )
                    )

                    # Clip refinement to its padded prompt box.
                    x1, y1, x2, y2 = (
                        item["box"]
                        .astype(int)
                        .tolist()
                    )

                    clipped = np.zeros_like(
                        sam_mask,
                        dtype=bool,
                    )

                    clipped[
                        y1 : y2 + 1,
                        x1 : x2 + 1,
                    ] = sam_mask[
                        y1 : y2 + 1,
                        x1 : x2 + 1,
                    ]

                    refined_mask |= clipped

        agreement = mask_iou(
            teacher_mask,
            refined_mask,
        )

        teacher_fg = float(
            teacher_mask.mean()
        )
        refined_fg = float(
            refined_mask.mean()
        )

        canvas = make_visualization(
            image,
            teacher_mask,
            refined_mask,
            components,
        )

        out_path = (
            image_output_dir
            / f"{stem}_teacher_vs_sam2.jpg"
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
            "native_h": orig_h,
            "native_w": orig_w,
            "teacher_fg_ratio": teacher_fg,
            "sam2_refined_fg_ratio": refined_fg,
            "teacher_refined_iou": agreement,
            "prompt_components": len(components),
            "mean_sam2_quality_score": (
                float(np.mean(sam_scores))
                if sam_scores
                else ""
            ),
        })

    csv_path = (
        output_dir
        / "diagnostic_stats.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 88)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 88)
    print(
        "Visualizations:",
        image_output_dir,
    )
    print(
        "Statistics     :",
        csv_path,
    )
    print("\nPanel order:")
    print(
        "Original | EMA Teacher coarse mask | "
        "Auto prompts | SAM2 refined mask | Refined overlay"
    )
    print("\nIMPORTANT:")
    print(
        "These are unlabeled training images. "
        "No Ground Truth was used to create the SAM2 prompts."
    )
    print(
        "Do not interpret teacher_refined_iou as accuracy; "
        "it only measures how much SAM2 changed the teacher mask."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
