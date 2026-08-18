from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "sam2_repo" / "checkpoints" / "sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
TEST_DIR = ROOT / "datasets" / "Crack500" / "testcrop"
OUT_DIR = ROOT / "results" / "sam2_sanity"


def find_nonempty_pair():
    for image_path in sorted(TEST_DIR.glob("*.jpg")):
        mask_path = image_path.with_suffix(".png")
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and (mask > 127).any():
            return image_path, mask_path
    raise RuntimeError("No non-empty CRACK500 test mask found.")


def bbox_from_mask(mask, pad=8):
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        raise RuntimeError("Mask is empty.")

    h, w = mask.shape
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w - 1, int(xs.max()) + pad)
    y2 = min(h - 1, int(ys.max()) + pad)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def binary_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = np.logical_or(pred, gt).sum()

    dice = (2.0 * inter) / max(pred_sum + gt_sum, 1)
    iou = inter / max(union, 1)
    return float(dice), float(iou)


def main():
    print("=" * 76)
    print("SAM2.1 SMALL SINGLE-IMAGE SANITY TEST")
    print("=" * 76)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}\n"
            "Download sam2.1_hiera_small.pt first."
        )

    image_path, mask_path = find_nonempty_pair()

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None or gt is None:
        raise RuntimeError("Failed to read test image/mask.")

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    box = bbox_from_mask(gt)

    print("Image      :", image_path.name)
    print("Native size:", image.shape[:2])
    print("GT box XYXY:", box.tolist())
    print("Checkpoint :", CHECKPOINT)
    print("CUDA       :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU        :", torch.cuda.get_device_name(0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nBuilding SAM2.1 Hiera Small...")
    sam2_model = build_sam2(
        MODEL_CFG,
        str(CHECKPOINT),
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )

    print("Running prediction...")
    with torch.inference_mode(), amp_context:
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False,
        )

    pred = masks[0].astype(bool)
    gt_binary = gt > 127

    dice, iou = binary_metrics(pred, gt_binary)

    print("\nSAM2 prediction finished.")
    print("Predicted quality score:", float(scores[0]))
    print(f"Sanity Dice vs GT      : {dice:.4f}")
    print(f"Sanity IoU vs GT       : {iou:.4f}")
    print(
        "NOTE: GT is used only to create this diagnostic box prompt. "
        "These numbers are NOT paper results."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_vis = (gt_binary.astype(np.uint8) * 255)
    pred_vis = (pred.astype(np.uint8) * 255)

    gt_rgb = np.repeat(gt_vis[..., None], 3, axis=2)
    pred_rgb = np.repeat(pred_vis[..., None], 3, axis=2)

    input_vis = image.copy()
    x1, y1, x2, y2 = box.astype(int).tolist()
    cv2.rectangle(input_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    overlay = image.copy()
    overlay[pred] = (
        0.45 * overlay[pred]
        + 0.55 * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)

    canvas = np.concatenate(
        [input_vis, gt_rgb, pred_rgb, overlay],
        axis=1,
    )

    out_path = OUT_DIR / f"{image_path.stem}_sam2_sanity.jpg"
    cv2.imwrite(
        str(out_path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )

    print("\nSaved visualization:")
    print(out_path)
    print("\nLayout:")
    print("Input+GT-derived box | Ground Truth | SAM2 mask | SAM2 overlay")
    print("=" * 76)


if __name__ == "__main__":
    main()
