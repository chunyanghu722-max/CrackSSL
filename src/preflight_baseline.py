from collections import Counter
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from crack500_data import Crack500Dataset, denormalize_image
from baseline_utils import build_model, bce_dice_loss, set_seed


def inspect_sizes(folder):
    sizes = Counter()
    for p in sorted(Path(folder).glob("*.jpg")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        sizes[(h, w)] += 1
    return sizes


def inspect_masks(folder, limit=30):
    stats = Counter()
    foreground_ratios = []
    for p in sorted(Path(folder).glob("*.png"))[:limit]:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        vals = tuple(np.unique(m).tolist())
        stats[vals] += 1
        foreground_ratios.append(float((m > 127).mean()))
    return stats, foreground_ratios


def save_preflight_examples(batch, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(4, batch["image"].shape[0])
    for i in range(n):
        img = denormalize_image(batch["image"][i])
        gt = (batch["mask"][i, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        gt_rgb = np.repeat(gt[..., None], 3, axis=2)

        overlay = img.copy()
        crack = gt > 0
        overlay[crack] = (
            0.4 * overlay[crack] + 0.6 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)

        canvas = np.concatenate([img, gt_rgb, overlay], axis=1)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"sample_{i+1}.jpg"), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/Crack500")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    set_seed(42)

    root = Path(args.data_root)
    print("=" * 78)
    print("CRACK500 BASELINE PREFLIGHT")
    print("=" * 78)

    for split in ["traincrop", "valcrop", "testcrop"]:
        sizes = inspect_sizes(root / split)
        print(f"\n[{split}] unique native image sizes:")
        for size, count in sizes.most_common(10):
            print(f"  HxW={size[0]}x{size[1]} : {count} images")
        if len(sizes) > 10:
            print(f"  ... {len(sizes)} unique sizes total")

    mask_stats, fg = inspect_masks(root / "traincrop", limit=30)
    print("\n[traincrop] mask value patterns in first 30 masks:")
    for vals, count in mask_stats.items():
        print(f"  values={vals} : {count}")
    if fg:
        print(
            f"Foreground ratio (first 30): mean={np.mean(fg):.4f}, "
            f"min={np.min(fg):.4f}, max={np.max(fg):.4f}"
        )

    ds = Crack500Dataset(
        root_dir=root,
        split="traincrop",
        height=args.height,
        width=args.width,
        train=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # maximum Windows compatibility for preflight
        pin_memory=True,
    )
    batch = next(iter(loader))

    print("\nProcessed batch:")
    print("  image:", tuple(batch["image"].shape), batch["image"].dtype)
    print("  mask :", tuple(batch["mask"].shape), batch["mask"].dtype)
    print("  mask unique:", torch.unique(batch["mask"]).tolist())

    save_preflight_examples(
        batch,
        "results/baseline_100pct/preflight_samples",
    )
    print(
        "Saved visual checks to: "
        "results/baseline_100pct/preflight_samples"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nBuilding DeepLabV3+ ResNet50 (ImageNet pretrained)...")
    print("First run may download ResNet50 encoder weights.")
    model = build_model(pretrained=True).to(device)
    model.train()

    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=(device.type == "cuda"),
    ):
        logits = model(images)
        loss, bce, dice_loss = bce_dice_loss(logits, masks)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print("\nOne complete train step: OK")
    print("  output:", tuple(logits.shape))
    print(f"  total loss: {loss.item():.6f}")
    print(f"  BCE loss  : {bce.item():.6f}")
    print(f"  Dice loss : {dice_loss.item():.6f}")

    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"  peak allocated VRAM: {allocated:.2f} GB")
        print(f"  peak reserved VRAM : {reserved:.2f} GB")

    print("\n" + "=" * 78)
    print("PREFLIGHT PASSED")
    print("If native image size is 360x640 and this step is OK,")
    print("you can start the full baseline training.")
    print("=" * 78)


if __name__ == "__main__":
    main()
