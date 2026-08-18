from pathlib import Path
from contextlib import nullcontext
import argparse
import csv
import json

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm

from baseline_utils import build_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from diagnose_conservative_sam2 import (
    teacher_preprocess,
    restore_probability,
    extract_components,
    select_safe_candidate,
)


def cldice_one(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    tprec = (
        np.logical_and(skel_pred, gt).sum()
        / max(skel_pred.sum(), 1)
    )
    tsens = (
        np.logical_and(skel_gt, pred).sum()
        / max(skel_gt.sum(), 1)
    )

    if tprec + tsens <= 0:
        return 0.0

    return float(
        2.0 * tprec * tsens
        / (tprec + tsens)
    )


class GlobalMeter:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.cldice_sum = 0.0
        self.count = 0

    def update(self, pred, gt):
        pred = np.asarray(pred, dtype=bool)
        gt = np.asarray(gt, dtype=bool)

        self.tp += int(
            np.logical_and(pred, gt).sum()
        )
        self.fp += int(
            np.logical_and(pred, ~gt).sum()
        )
        self.fn += int(
            np.logical_and(~pred, gt).sum()
        )
        self.tn += int(
            np.logical_and(~pred, ~gt).sum()
        )

        self.cldice_sum += cldice_one(
            pred,
            gt,
        )
        self.count += 1

    def compute(self):
        eps = 1e-8

        dice = (
            2.0 * self.tp
            / (
                2.0 * self.tp
                + self.fp
                + self.fn
                + eps
            )
        )

        iou = (
            self.tp
            / (
                self.tp
                + self.fp
                + self.fn
                + eps
            )
        )

        precision = (
            self.tp
            / (
                self.tp
                + self.fp
                + eps
            )
        )

        recall = (
            self.tp
            / (
                self.tp
                + self.fn
                + eps
            )
        )

        accuracy = (
            self.tp + self.tn
        ) / (
            self.tp
            + self.tn
            + self.fp
            + self.fn
            + eps
        )

        return {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "cldice": float(
                self.cldice_sum
                / max(self.count, 1)
            ),
            "accuracy": float(accuracy),
        }


def per_image_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    dice = (
        2.0 * tp
        / max(
            2.0 * tp + fp + fn,
            1,
        )
    )

    iou = (
        tp
        / max(
            tp + fp + fn,
            1,
        )
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "cldice": cldice_one(
            pred,
            gt,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher-checkpoint",
        default="results/ema_seed42_10pct/checkpoints/best.pt",
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
        "--output",
        default="results/sam2_conservative_val_10pct",
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

    sam2_model = build_sam2(
        args.sam2_config,
        args.sam2_checkpoint,
        device=str(device),
    )

    predictor = SAM2ImagePredictor(
        sam2_model
    )

    val_dir = (
        Path(args.data_root)
        / "valcrop"
    )

    image_paths = sorted(
        val_dir.glob("*.jpg")
    )

    if not image_paths:
        raise RuntimeError(
            f"No validation JPGs found in {val_dir}"
        )

    print("=" * 96)
    print("VALIDATION: EMA TEACHER vs CONSERVATIVE SAM2 REFINEMENT")
    print("=" * 96)
    print("Validation images :", len(image_paths))
    print("Teacher checkpoint:", args.teacher_checkpoint)
    print("Teacher threshold :", args.teacher_threshold)
    print("Box pad           :", args.box_pad)
    print("Trust radius      :", args.trust_radius)
    print("IMPORTANT         : GT is used ONLY for final evaluation, NEVER for SAM2 prompts.")
    print("=" * 96)

    teacher_meter = GlobalMeter()
    refined_meter = GlobalMeter()

    rows = []

    total_components = 0
    total_accepted = 0
    total_rejected = 0

    improved_dice_images = 0
    worsened_dice_images = 0
    equal_dice_images = 0

    for image_path in tqdm(
        image_paths,
        desc="Validate SAM2",
    ):
        mask_path = (
            image_path.with_suffix(".png")
        )

        if not mask_path.exists():
            raise RuntimeError(
                f"Missing GT mask: {mask_path}"
            )

        bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        gt_u8 = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if bgr is None or gt_u8 is None:
            raise RuntimeError(
                f"Cannot read pair: {image_path.name}"
            )

        image = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        gt = (
            gt_u8 > 127
        )

        h, w = image.shape[:2]

        teacher_tensor, params = (
            teacher_preprocess(
                image,
                target_h,
                target_w,
            )
        )

        teacher_tensor = (
            teacher_tensor.to(
                device,
                non_blocking=True,
            )
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
                    teacher_tensor
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

        refined = teacher_mask.copy()

        accepted = 0
        rejected = 0

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

                    for item in components:
                        pos = item[
                            "positive_points"
                        ]

                        neg = item[
                            "negative_points"
                        ]

                        if len(neg) > 0:
                            points = (
                                np.concatenate(
                                    [pos, neg],
                                    axis=0,
                                )
                            )

                            labels = (
                                np.concatenate(
                                    [
                                        np.ones(
                                            len(pos),
                                            dtype=np.int32,
                                        ),
                                        np.zeros(
                                            len(neg),
                                            dtype=np.int32,
                                        ),
                                    ],
                                    axis=0,
                                )
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
                            rejected += 1
                            continue

                        accepted += 1

                        refined[
                            item["component"]
                        ] = False

                        refined |= chosen[
                            "mask"
                        ]

        total_components += len(
            components
        )
        total_accepted += accepted
        total_rejected += rejected

        teacher_meter.update(
            teacher_mask,
            gt,
        )

        refined_meter.update(
            refined,
            gt,
        )

        teacher_img = (
            per_image_metrics(
                teacher_mask,
                gt,
            )
        )

        refined_img = (
            per_image_metrics(
                refined,
                gt,
            )
        )

        delta_dice = (
            refined_img["dice"]
            - teacher_img["dice"]
        )

        if delta_dice > 1e-6:
            improved_dice_images += 1
        elif delta_dice < -1e-6:
            worsened_dice_images += 1
        else:
            equal_dice_images += 1

        rows.append({
            "stem": image_path.stem,
            "components": len(components),
            "accepted_components": accepted,
            "rejected_components": rejected,

            "teacher_dice": teacher_img["dice"],
            "refined_dice": refined_img["dice"],
            "delta_dice": delta_dice,

            "teacher_iou": teacher_img["iou"],
            "refined_iou": refined_img["iou"],
            "delta_iou": (
                refined_img["iou"]
                - teacher_img["iou"]
            ),

            "teacher_cldice": teacher_img["cldice"],
            "refined_cldice": refined_img["cldice"],
            "delta_cldice": (
                refined_img["cldice"]
                - teacher_img["cldice"]
            ),
        })

    teacher_metrics = (
        teacher_meter.compute()
    )

    refined_metrics = (
        refined_meter.compute()
    )

    delta = {
        key: (
            refined_metrics[key]
            - teacher_metrics[key]
        )
        for key in teacher_metrics
    }

    acceptance_rate = (
        total_accepted
        / max(total_components, 1)
    )

    out_dir = Path(
        args.output
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir
        / "per_image_metrics.csv"
    )

    with csv_path.open(
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

    summary = {
        "num_validation_images": len(
            image_paths
        ),
        "teacher": teacher_metrics,
        "conservative_sam2": refined_metrics,
        "delta_refined_minus_teacher": delta,
        "total_components": total_components,
        "accepted_components": total_accepted,
        "rejected_components": total_rejected,
        "component_acceptance_rate": acceptance_rate,
        "images_with_dice_improvement": improved_dice_images,
        "images_with_dice_degradation": worsened_dice_images,
        "images_with_equal_dice": equal_dice_images,
        "config": vars(args),
    }

    json_path = (
        out_dir
        / "summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 96)
    print("VALIDATION RESULTS")
    print("=" * 96)

    print("\nEMA Teacher:")
    for key, value in teacher_metrics.items():
        print(
            f"  {key:>10s}: {value:.6f}"
        )

    print("\nConservative SAM2:")
    for key, value in refined_metrics.items():
        print(
            f"  {key:>10s}: {value:.6f}"
        )

    print("\nDelta (SAM2 - Teacher):")
    for key, value in delta.items():
        print(
            f"  {key:>10s}: {value:+.6f}"
        )

    print("\nComponent safety:")
    print(
        "  total components   :",
        total_components,
    )
    print(
        "  accepted components:",
        total_accepted,
    )
    print(
        "  rejected components:",
        total_rejected,
    )
    print(
        "  acceptance rate    :",
        f"{acceptance_rate:.3f}",
    )

    print("\nPer-image Dice:")
    print(
        "  improved :",
        improved_dice_images,
    )
    print(
        "  worsened :",
        worsened_dice_images,
    )
    print(
        "  unchanged:",
        equal_dice_images,
    )

    print("\nSaved:")
    print(" ", json_path)
    print(" ", csv_path)

    print("\nDecision rule:")
    print(
        "If Dice/IoU/clDice improve overall without a large number "
        "of severely degraded images, proceed to batch unlabeled pseudo-label generation."
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
