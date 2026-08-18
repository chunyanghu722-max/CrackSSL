from pathlib import Path
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crack500_data import Crack500Dataset
from baseline_utils import (
    BinarySegmentationMeter,
    append_csv,
    bce_dice_loss,
    build_model,
    save_json,
    set_seed,
)


def make_loader(ds, batch_size, workers, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        generator=g,
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
    meter = BinarySegmentationMeter(threshold=0.5)
    total_loss = 0.0
    seen = 0

    bar = tqdm(loader, desc="Val", leave=False)
    for batch in bar:
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
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/baseline_100pct")
    parser.add_argument("--resume", default=None, help="Path to last.pt to resume training")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.output)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    save_json(config, out_dir / "config.json")

    train_ds = Crack500Dataset(
        args.data_root, "traincrop",
        args.height, args.width, train=True
    )
    val_ds = Crack500Dataset(
        args.data_root, "valcrop",
        args.height, args.width, train=False
    )

    train_loader = make_loader(
        train_ds, args.batch_size, args.workers, True, args.seed
    )
    val_loader = make_loader(
        val_ds, args.batch_size, args.workers, False, args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("DeepLabV3+ ResNet50 | CRACK500 | 100% labeled baseline")
    print("=" * 80)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Train samples:", len(train_ds))
    print("Val samples  :", len(val_ds))
    print(f"Input        : {args.height}x{args.width}")
    print("Batch size   :", args.batch_size)
    print("Workers      :", args.workers)
    print("Epochs       :", args.epochs)
    print("Seed         :", args.seed)
    print("=" * 80)

    model = build_model(pretrained=(args.resume is None)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=str(log_dir))

    best_dice = -1.0
    history = []
    start_epoch = 1
    start_time = time.time()

    csv_path = out_dir / "history.csv"

    # Recover existing history if present.
    if csv_path.exists():
        import csv
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if k == "epoch":
                        parsed[k] = int(float(v))
                    else:
                        try:
                            parsed[k] = float(v)
                        except Exception:
                            parsed[k] = v
                history.append(parsed)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"Resuming from: {resume_path}")
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_val_dice", -1.0))

        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            try:
                best_checkpoint = torch.load(
                    best_path,
                    map_location="cpu",
                    weights_only=False,
                )
                best_dice = max(
                    best_dice,
                    float(best_checkpoint.get("best_val_dice", -1.0)),
                )
            except Exception:
                pass

        print(f"Resume epoch : {start_epoch}")
        print(f"Best Dice so far: {best_dice:.6f}")

        if start_epoch > args.epochs:
            print("Checkpoint already reached the requested total epochs.")
            return
    else:
        # Fresh run: clear old history file to avoid mixing experiments.
        if csv_path.exists():
            csv_path.unlink()
        history = []

    for epoch in range(start_epoch, args.epochs + 1):
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
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

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
            f"clDice {val['cldice']:.4f}"
            f"{mark}"
        )

        save_curves(history, out_dir)

    writer.close()

    summary = {
        "best_val_dice": best_dice,
        "total_minutes": (time.time() - start_time) / 60.0,
        "best_checkpoint": str(ckpt_dir / "best.pt"),
    }
    save_json(summary, out_dir / "training_summary.json")

    print("\nTraining finished.")
    print("Best validation Dice:", f"{best_dice:.6f}")
    print("Best checkpoint:", ckpt_dir / "best.pt")
    print("Next command:")
    print("python src\\evaluate_baseline.py")


if __name__ == "__main__":
    main()
