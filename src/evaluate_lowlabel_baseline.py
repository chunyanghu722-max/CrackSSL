from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crack500_subset_data import Crack500SubsetDataset, denormalize_image
from baseline_utils import (
    BinarySegmentationMeter,
    bce_dice_loss,
    build_model,
    save_json,
)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meter = BinarySegmentationMeter(threshold=0.5, compute_cldice=True)

    total_loss = 0.0
    seen = 0

    for batch in tqdm(loader, desc="Test"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss, _, _ = bce_dice_loss(logits, masks)

        bs = images.size(0)
        total_loss += loss.item() * bs
        seen += bs
        meter.update(logits.float(), masks)

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics


@torch.no_grad()
def save_prediction_examples(model, loader, device, out_dir, max_images=12):
    """
    Save:
    original | ground truth | prediction | prediction overlay
    """
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for batch in loader:
        images_gpu = batch["image"].to(device, non_blocking=True)

        logits = model(images_gpu)
        probs = torch.sigmoid(logits).cpu()
        preds = probs >= 0.5

        for i in range(images_gpu.size(0)):
            image = denormalize_image(batch["image"][i])

            gt = (
                batch["mask"][i, 0].cpu().numpy() > 0.5
            ).astype(np.uint8) * 255

            pred = (
                preds[i, 0].numpy().astype(np.uint8)
            ) * 255

            gt_rgb = np.repeat(gt[..., None], 3, axis=2)
            pred_rgb = np.repeat(pred[..., None], 3, axis=2)

            overlay = image.copy()
            crack = pred > 0
            overlay[crack] = (
                0.4 * overlay[crack]
                + 0.6 * np.array([255, 0, 0], dtype=np.float32)
            ).astype(np.uint8)

            canvas = np.concatenate(
                [image, gt_rgb, pred_rgb, overlay],
                axis=1,
            )

            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

            name = batch["name"][i]
            out_path = out_dir / f"{saved+1:02d}_{name}.jpg"
            cv2.imwrite(str(out_path), canvas)

            saved += 1
            if saved >= max_images:
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="datasets/Crack500")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    cfg = ckpt["config"]

    ds = Crack500SubsetDataset(
        args.data_root,
        split="testcrop",
        list_file=None,
        height=int(cfg["height"]),
        width=int(cfg["width"]),
        train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    metrics = evaluate(model, loader, device)

    print("\n" + "=" * 72)
    print("CRACK500 LOW-LABEL TEST RESULTS")
    print("=" * 72)
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.6f}")
    print("=" * 72)

    out_dir = Path(cfg["output"])

    save_json(metrics, out_dir / "test_metrics.json")

    pred_dir = out_dir / "test_predictions"
    save_prediction_examples(
        model,
        loader,
        device,
        pred_dir,
        max_images=args.max_images,
    )

    print("Saved:", out_dir / "test_metrics.json")
    print("Saved prediction examples:", pred_dir)


if __name__ == "__main__":
    main()
