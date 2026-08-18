from pathlib import Path
import argparse
import time
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from crack500_subset_data import Crack500SubsetDataset
from baseline_utils import (
    BinarySegmentationMeter,
    append_csv,
    bce_dice_loss,
    build_model,
    save_json,
    set_seed,
)


def make_train_loader(ds, batch_size, workers, epoch_samples, seed):
    # Sample WITH replacement so every low-label setting has the same
    # number of training examples/updates per epoch as the 100% baseline.
    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = RandomSampler(
        ds,
        replacement=True,
        num_samples=epoch_samples,
        generator=generator,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
    )


def make_eval_loader(ds, batch_size, workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
    )


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    seen = 0

    bar = tqdm(loader, desc="Train", leave=False)
    for batch in bar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss, _, _ = bce_dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        seen += bs
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(seen, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = BinarySegmentationMeter(threshold=0.5, compute_cldice=True)

    total_loss = 0.0
    seen = 0

    for batch in tqdm(loader, desc="Val", leave=False):
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


def save_curves(history, out_dir):
    if not history:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [x["epoch"] for x in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [x["train_loss"] for x in history], label="Train loss")
    plt.plot(epochs, [x["val_loss"] for x in history], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [x["val_dice"] for x in history], label="Dice")
    plt.plot(epochs, [x["val_iou"] for x in history], label="IoU")
    plt.plot(epochs, [x["val_cldice"] for x in history], label="clDice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "metric_curve.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/Crack500")
    parser.add_argument("--split-list", required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch-samples", type=int, default=1896)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    save_json(config, out_dir / "config.json")

    train_ds = Crack500SubsetDataset(
        args.data_root,
        split="traincrop",
        list_file=args.split_list,
        height=args.height,
        width=args.width,
        train=True,
    )
    val_ds = Crack500SubsetDataset(
        args.data_root,
        split="valcrop",
        list_file=None,
        height=args.height,
        width=args.width,
        train=False,
    )

    train_loader = make_train_loader(
        train_ds,
        args.batch_size,
        args.workers,
        args.epoch_samples,
        args.seed,
    )
    val_loader = make_eval_loader(
        val_ds,
        args.batch_size,
        args.workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print("CRACK500 LOW-LABEL SUPERVISED BASELINE")
    print("=" * 84)
    print("Device        :", device)
    if device.type == "cuda":
        print("GPU           :", torch.cuda.get_device_name(0))
    print("Unique labeled:", len(train_ds))
    print("Epoch samples :", args.epoch_samples)
    print("Updates/epoch :", len(train_loader))
    print("Val samples   :", len(val_ds))
    print("Split list    :", args.split_list)
    print("Input         :", f"{args.height}x{args.width}")
    print("Epochs        :", args.epochs)
    print("Seed          :", args.seed)
    print("=" * 84)

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=str(log_dir))

    best_dice = -1.0
    history = []
    csv_path = out_dir / "history.csv"
    if csv_path.exists():
        csv_path.unlink()

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device
        )
        val = validate(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_cldice": val["cldice"],
            "val_accuracy": val["accuracy"],
            "epoch_minutes": (time.time() - epoch_start) / 60.0,
        }
        history.append(row)
        append_csv(csv_path, row)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val["loss"], epoch)
        writer.add_scalar("metric/dice", val["dice"], epoch)
        writer.add_scalar("metric/iou", val["iou"], epoch)
        writer.add_scalar("metric/precision", val["precision"], epoch)
        writer.add_scalar("metric/recall", val["recall"], epoch)
        writer.add_scalar("metric/cldice", val["cldice"], epoch)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_dice": best_dice,
            "config": config,
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")

        improved = val["dice"] > best_dice
        if improved:
            best_dice = val["dice"]
            checkpoint["best_val_dice"] = best_dice
            torch.save(checkpoint, ckpt_dir / "best.pt")

        mark = "  <-- BEST" if improved else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {train_loss:.4f} | val {val['loss']:.4f} | "
            f"Dice {val['dice']:.4f} | IoU {val['iou']:.4f} | "
            f"P {val['precision']:.4f} | R {val['recall']:.4f} | "
            f"clDice {val['cldice']:.4f}{mark}"
        )

        save_curves(history, out_dir)

    writer.close()

    summary = {
        "unique_labeled_samples": len(train_ds),
        "epoch_samples": args.epoch_samples,
        "best_val_dice": best_dice,
        "total_minutes": (time.time() - start_time) / 60.0,
        "best_checkpoint": str(ckpt_dir / "best.pt"),
    }
    save_json(summary, out_dir / "training_summary.json")

    print("\nTraining finished.")
    print("Best validation Dice:", f"{best_dice:.6f}")
    print("Best checkpoint:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
