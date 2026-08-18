from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crack500_data import Crack500Dataset, denormalize_image
from baseline_utils import (
    BinarySegmentationMeter,
    bce_dice_loss,
    build_model,
    save_json,
)


def save_prediction_examples(model, loader, device, out_dir, max_images=12):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]

            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            preds = probs >= 0.5

            for i in range(images.size(0)):
                img = denormalize_image(batch["image"][i])
                gt = (masks[i, 0].numpy() > 0.5).astype(np.uint8) * 255
                pr = preds[i, 0].numpy().astype(np.uint8) * 255

                gt_rgb = np.repeat(gt[..., None], 3, axis=2)
                pr_rgb = np.repeat(pr[..., None], 3, axis=2)

                overlay = img.copy()
                crack = pr > 0
                overlay[crack] = (
                    0.4 * overlay[crack]
                    + 0.6 * np.array([255, 0, 0], dtype=np.float32)
                ).astype(np.uint8)

                canvas = np.concatenate(
                    [img, gt_rgb, pr_rgb, overlay],
                    axis=1,
                )
                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

                name = batch["name"][i]
                cv2.imwrite(
                    str(out_dir / f"{saved+1:02d}_{name}.jpg"),
                    canvas,
                )
                saved += 1
                if saved >= max_images:
                    return


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meter = BinarySegmentationMeter(threshold=0.5)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/baseline_100pct/checkpoints/best.pt",
    )
    parser.add_argument("--data-root", default="datasets/Crack500")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    cfg = checkpoint["config"]
    height = int(cfg["height"])
    width = int(cfg["width"])

    ds = Crack500Dataset(
        args.data_root,
        "testcrop",
        height,
        width,
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])

    metrics = evaluate(model, loader, device)

    print("\n" + "=" * 70)
    print("CRACK500 TEST RESULTS")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.6f}")
    print("=" * 70)

    out_dir = Path(cfg["output"])
    save_json(metrics, out_dir / "test_metrics.json")
    save_prediction_examples(
        model,
        loader,
        device,
        out_dir / "test_predictions",
        max_images=12,
    )
    print("Saved:", out_dir / "test_metrics.json")
    print("Saved prediction examples:", out_dir / "test_predictions")


if __name__ == "__main__":
    main()
