from pathlib import Path
import argparse, copy, math

import matplotlib
matplotlib.use("Agg")

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from crack500_subset_data import Crack500SubsetDataset
from ema_data import Crack500UnlabeledPairDataset
from baseline_utils import (
    BinarySegmentationMeter,
    append_csv,
    bce_dice_loss,
    build_model,
    save_json,
    set_seed,
)


def rampup(epoch, length):
    if length <= 0:
        return 1.0
    x = max(0.0, min(float(epoch), float(length)))
    phase = 1.0 - x / float(length)
    return math.exp(-5.0 * phase * phase)


def rep_loader(ds, batch, workers, samples, seed):
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(
        ds,
        replacement=True,
        num_samples=samples,
        generator=g,
    )
    return DataLoader(
        ds,
        batch_size=batch,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )


def eval_loader(ds, batch, workers):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )


@torch.no_grad()
def update_teacher(teacher, student, decay, step):
    alpha = min(1.0 - 1.0 / (step + 1.0), decay)

    for t_param, s_param in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        t_param.data.mul_(alpha).add_(
            s_param.data,
            alpha=1.0 - alpha,
        )

    for t_buf, s_buf in zip(
        teacher.buffers(),
        student.buffers(),
    ):
        t_buf.copy_(s_buf)


def masked_soft_consistency(
    student_probs,
    teacher_probs,
    confidence_threshold,
):
    """
    Binary teacher confidence:
        confidence = max(p, 1-p)

    Only pixels with confidence >= threshold contribute to consistency.
    Teacher probabilities remain SOFT; only low-confidence locations are masked.
    """
    confidence = torch.maximum(
        teacher_probs,
        1.0 - teacher_probs,
    )

    keep = confidence >= confidence_threshold

    diff2 = (student_probs - teacher_probs).pow(2)

    kept_pixels = keep.sum()
    if kept_pixels.item() == 0:
        # Maintain graph for student without contributing a gradient.
        loss = diff2.sum() * 0.0
    else:
        loss = diff2[keep].mean()

    keep_ratio = keep.float().mean()

    teacher_binary = teacher_probs >= 0.5
    kept_positive = torch.logical_and(
        keep,
        teacher_binary,
    ).float().sum()

    kept_negative = torch.logical_and(
        keep,
        ~teacher_binary,
    ).float().sum()

    denom = kept_pixels.float().clamp_min(1.0)

    positive_fraction_among_kept = kept_positive / denom
    negative_fraction_among_kept = kept_negative / denom

    return (
        loss,
        keep_ratio.detach(),
        positive_fraction_among_kept.detach(),
        negative_fraction_among_kept.detach(),
    )


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    meter = BinarySegmentationMeter(
        threshold=0.5,
        compute_cldice=True,
    )

    loss_sum = 0.0
    seen = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss, _, _ = bce_dice_loss(
                logits,
                masks,
            )

        loss_sum += loss.item() * images.size(0)
        seen += images.size(0)

        meter.update(
            logits.float(),
            masks,
        )

    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(seen, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="datasets/Crack500",
    )
    parser.add_argument(
        "--labeled-list",
        required=True,
    )
    parser.add_argument(
        "--unlabeled-list",
        required=True,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--epoch-samples",
        type=int,
        default=1896,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--max-unsup-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--rampup-epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    if not (0.5 <= args.conf_threshold < 1.0):
        raise ValueError(
            "--conf-threshold should be in [0.5, 1.0)"
        )

    set_seed(args.seed)

    out_dir = Path(args.output)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        vars(args),
        out_dir / "config.json",
    )

    labeled_ds = Crack500SubsetDataset(
        args.data_root,
        split="traincrop",
        list_file=args.labeled_list,
        height=args.height,
        width=args.width,
        train=True,
    )

    unlabeled_ds = Crack500UnlabeledPairDataset(
        args.data_root,
        args.unlabeled_list,
        args.height,
        args.width,
    )

    val_ds = Crack500SubsetDataset(
        args.data_root,
        split="valcrop",
        list_file=None,
        height=args.height,
        width=args.width,
        train=False,
    )

    labeled_loader = rep_loader(
        labeled_ds,
        args.batch_size,
        args.workers,
        args.epoch_samples,
        args.seed,
    )

    unlabeled_loader = rep_loader(
        unlabeled_ds,
        args.batch_size,
        args.workers,
        args.epoch_samples,
        args.seed + 1,
    )

    val_loader = eval_loader(
        val_ds,
        args.batch_size,
        args.workers,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 88)
    print("CRACK500 EMA + CONFIDENCE FILTERING")
    print("=" * 88)
    print("Unique labeled      :", len(labeled_ds))
    print("Unique unlabeled    :", len(unlabeled_ds))
    print("Updates/epoch       :", len(labeled_loader))
    print("EMA decay           :", args.ema_decay)
    print("Confidence threshold:", args.conf_threshold)
    print("Ramp-up epochs      :", args.rampup_epochs)
    print("=" * 88)

    student = build_model(
        pretrained=True,
    ).to(device)

    teacher = copy.deepcopy(
        student,
    ).to(device)

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    teacher.eval()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    best_dice = -1.0
    global_step = 0

    history_path = out_dir / "history.csv"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        student.train()
        teacher.eval()

        unsup_weight = (
            args.max_unsup_weight
            * rampup(
                epoch - 1,
                args.rampup_epochs,
            )
        )

        sup_sum = 0.0
        con_sum = 0.0
        keep_sum = 0.0
        pos_sum = 0.0
        neg_sum = 0.0
        seen = 0

        iterator = zip(
            labeled_loader,
            unlabeled_loader,
        )

        bar = tqdm(
            iterator,
            total=min(
                len(labeled_loader),
                len(unlabeled_loader),
            ),
            desc="Train",
            leave=False,
        )

        for labeled_batch, unlabeled_batch in bar:
            x_l = labeled_batch["image"].to(
                device,
                non_blocking=True,
            )
            y_l = labeled_batch["mask"].to(
                device,
                non_blocking=True,
            )

            x_u_weak = unlabeled_batch["weak"].to(
                device,
                non_blocking=True,
            )
            x_u_strong = unlabeled_batch["strong"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True,
            )

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    teacher_logits = teacher(
                        x_u_weak
                    )

                teacher_probs = torch.sigmoid(
                    teacher_logits.float()
                )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                labeled_logits = student(
                    x_l
                )

                supervised_loss, _, _ = bce_dice_loss(
                    labeled_logits,
                    y_l,
                )

                student_unlabeled_logits = student(
                    x_u_strong
                )

                student_probs = torch.sigmoid(
                    student_unlabeled_logits.float()
                )

                (
                    consistency_loss,
                    keep_ratio,
                    positive_fraction,
                    negative_fraction,
                ) = masked_soft_consistency(
                    student_probs,
                    teacher_probs,
                    args.conf_threshold,
                )

                total_loss = (
                    supervised_loss
                    + unsup_weight
                    * consistency_loss
                )

            scaler.scale(
                total_loss
            ).backward()

            scaler.step(
                optimizer
            )
            scaler.update()

            global_step += 1

            update_teacher(
                teacher,
                student,
                args.ema_decay,
                global_step,
            )

            batch_size = x_l.size(0)

            sup_sum += (
                supervised_loss.item()
                * batch_size
            )
            con_sum += (
                consistency_loss.item()
                * batch_size
            )
            keep_sum += (
                keep_ratio.item()
                * batch_size
            )
            pos_sum += (
                positive_fraction.item()
                * batch_size
            )
            neg_sum += (
                negative_fraction.item()
                * batch_size
            )
            seen += batch_size

            bar.set_postfix(
                sup=f"{supervised_loss.item():.4f}",
                con=f"{consistency_loss.item():.5f}",
                keep=f"{keep_ratio.item():.3f}",
            )

        scheduler.step()

        validation = validate(
            teacher,
            val_loader,
            device,
        )

        sup_avg = sup_sum / max(seen, 1)
        con_avg = con_sum / max(seen, 1)
        keep_avg = keep_sum / max(seen, 1)
        positive_avg = pos_sum / max(seen, 1)
        negative_avg = neg_sum / max(seen, 1)

        row = {
            "epoch": epoch,
            "unsup_weight": unsup_weight,
            "confidence_threshold": args.conf_threshold,
            "train_sup_loss": sup_avg,
            "train_filtered_consistency_loss": con_avg,
            "pseudo_keep_ratio": keep_avg,
            "pseudo_positive_fraction_among_kept": positive_avg,
            "pseudo_negative_fraction_among_kept": negative_avg,
            "val_loss": validation["loss"],
            "val_dice": validation["dice"],
            "val_iou": validation["iou"],
            "val_precision": validation["precision"],
            "val_recall": validation["recall"],
            "val_cldice": validation["cldice"],
            "val_accuracy": validation["accuracy"],
        }

        append_csv(
            history_path,
            row,
        )

        state = {
            "epoch": epoch,
            "student_model": student.state_dict(),
            "teacher_model": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_dice": best_dice,
            "global_step": global_step,
            "config": vars(args),
        }

        torch.save(
            state,
            checkpoint_dir / "last.pt",
        )

        improved = (
            validation["dice"]
            > best_dice
        )

        if improved:
            best_dice = validation["dice"]
            state["best_val_dice"] = best_dice

            torch.save(
                state,
                checkpoint_dir / "best.pt",
            )

        best_mark = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"sup {sup_avg:.4f} | "
            f"con {con_avg:.5f} | "
            f"wu {unsup_weight:.3f} | "
            f"keep {keep_avg:.3f} | "
            f"Dice {validation['dice']:.4f} | "
            f"IoU {validation['iou']:.4f} | "
            f"P {validation['precision']:.4f} | "
            f"R {validation['recall']:.4f} | "
            f"clDice {validation['cldice']:.4f}"
            f"{best_mark}"
        )

    print("\nConfidence-filtered EMA training finished.")
    print(
        "Best teacher validation Dice:",
        f"{best_dice:.6f}",
    )
    print(
        "Best checkpoint:",
        checkpoint_dir / "best.pt",
    )


if __name__ == "__main__":
    main()
