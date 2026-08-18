from pathlib import Path
import argparse
import copy
import math

import torch
import torch.nn.functional as F
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
from structural_topology_utils import centerline_node_consistency


def rampup(epoch, length):
    if length <= 0:
        return 1.0
    x = max(0.0, min(float(epoch), float(length)))
    phase = 1.0 - x / float(length)
    return math.exp(-5.0 * phase * phase)


def replacement_loader(dataset, batch_size, workers, epoch_samples, seed):
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=epoch_samples,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
    )


def eval_loader(dataset, batch_size, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
    )


@torch.no_grad()
def update_teacher(teacher, student, decay, global_step):
    alpha = min(1.0 - 1.0 / float(global_step + 1), decay)
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = BinarySegmentationMeter(threshold=0.5, compute_cldice=True)
    loss_sum = 0.0
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
        loss_sum += loss.item() * bs
        seen += bs
        meter.update(logits.float(), masks)

    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(seen, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/Crack500")
    parser.add_argument("--labeled-list", required=True)
    parser.add_argument("--unlabeled-list", required=True)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch-samples", type=int, default=1896)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--max-unsup-weight", type=float, default=1.0)
    parser.add_argument("--rampup-epochs", type=int, default=20)

    # Frozen Step 11 centerline weight.
    parser.add_argument("--skel-weight", type=float, default=0.05)
    # New Step 13 structural-node weight.
    parser.add_argument("--node-weight", type=float, default=0.02)
    parser.add_argument("--skel-iters", type=int, default=10)
    parser.add_argument("--topo-downsample", type=int, default=2)
    parser.add_argument(
        "--node-radius",
        type=int,
        default=2,
        help="Endpoint/junction matching tolerance on the topology map.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.skel_weight < 0 or args.node_weight < 0:
        raise ValueError("Topology weights must be >= 0.")
    if args.skel_iters < 1 or args.topo_downsample < 1 or args.node_radius < 0:
        raise ValueError("Invalid topology configuration.")

    set_seed(args.seed)
    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), output_dir / "config.json")

    labeled_dataset = Crack500SubsetDataset(
        args.data_root,
        split="traincrop",
        list_file=args.labeled_list,
        height=args.height,
        width=args.width,
        train=True,
    )
    unlabeled_dataset = Crack500UnlabeledPairDataset(
        args.data_root,
        args.unlabeled_list,
        args.height,
        args.width,
    )
    validation_dataset = Crack500SubsetDataset(
        args.data_root,
        split="valcrop",
        list_file=None,
        height=args.height,
        width=args.width,
        train=False,
    )

    labeled_loader = replacement_loader(
        labeled_dataset,
        args.batch_size,
        args.workers,
        args.epoch_samples,
        args.seed,
    )
    unlabeled_loader = replacement_loader(
        unlabeled_dataset,
        args.batch_size,
        args.workers,
        args.epoch_samples,
        args.seed + 1,
    )
    validation_loader = eval_loader(
        validation_dataset,
        args.batch_size,
        args.workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 104)
    print("CRACK500 EMA + CENTERLINE + STRUCTURAL-NODE CONSISTENCY")
    print("=" * 104)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Unique labeled      :", len(labeled_dataset))
    print("Unique unlabeled    :", len(unlabeled_dataset))
    print("Updates/epoch       :", len(labeled_loader))
    print("EMA decay           :", args.ema_decay)
    print("Skeleton weight     :", args.skel_weight)
    print("Node weight         :", args.node_weight)
    print("Skeleton iterations :", args.skel_iters)
    print("Topology downsample :", args.topo_downsample)
    print("Node tolerance      :", args.node_radius, "px on topology map")
    print("Ramp-up epochs      :", args.rampup_epochs)
    print("=" * 104)

    student = build_model(pretrained=True).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_validation_dice = -1.0
    global_step = 0
    history_path = output_dir / "history.csv"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(1, args.epochs + 1):
        student.train()
        teacher.eval()
        wu = args.max_unsup_weight * rampup(epoch - 1, args.rampup_epochs)

        sup_sum = con_sum = skel_sum = node_sum = 0.0
        scl_sum = end_sum = junc_sum = total_sum = 0.0
        seen = 0

        progress = tqdm(
            zip(labeled_loader, unlabeled_loader),
            total=min(len(labeled_loader), len(unlabeled_loader)),
            desc="Train",
            leave=False,
        )

        for labeled_batch, unlabeled_batch in progress:
            xl = labeled_batch["image"].to(device, non_blocking=True)
            yl = labeled_batch["mask"].to(device, non_blocking=True)
            uw = unlabeled_batch["weak"].to(device, non_blocking=True)
            us = unlabeled_batch["strong"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    teacher_logits = teacher(uw)
                teacher_prob = torch.sigmoid(teacher_logits.float())

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                labeled_logits = student(xl)
                supervised_loss, _, _ = bce_dice_loss(labeled_logits, yl)
                student_unlabeled_logits = student(us)

            # Topology branch in FP32 for numerical stability.
            student_prob = torch.sigmoid(student_unlabeled_logits.float())
            consistency_loss = F.mse_loss(student_prob, teacher_prob)

            (
                skeleton_loss,
                soft_cldice_score,
                node_loss,
                endpoint_score,
                junction_score,
                _,
            ) = centerline_node_consistency(
                student_prob,
                teacher_prob,
                skeleton_iterations=args.skel_iters,
                downsample=args.topo_downsample,
                node_radius=args.node_radius,
            )

            unsupervised_loss = (
                consistency_loss
                + args.skel_weight * skeleton_loss
                + args.node_weight * node_loss
            )
            total_loss = supervised_loss + wu * unsupervised_loss

            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite loss detected.")

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            update_teacher(teacher, student, args.ema_decay, global_step)

            bs = xl.size(0)
            sup_sum += supervised_loss.item() * bs
            con_sum += consistency_loss.item() * bs
            skel_sum += skeleton_loss.item() * bs
            node_sum += node_loss.item() * bs
            scl_sum += soft_cldice_score.item() * bs
            end_sum += endpoint_score.item() * bs
            junc_sum += junction_score.item() * bs
            total_sum += total_loss.item() * bs
            seen += bs

            progress.set_postfix(
                sup=f"{supervised_loss.item():.4f}",
                con=f"{consistency_loss.item():.5f}",
                skel=f"{skeleton_loss.item():.4f}",
                node=f"{node_loss.item():.4f}",
            )

        scheduler.step()
        val = validate(teacher, validation_loader, device)

        sup_avg = sup_sum / max(seen, 1)
        con_avg = con_sum / max(seen, 1)
        skel_avg = skel_sum / max(seen, 1)
        node_avg = node_sum / max(seen, 1)
        scl_avg = scl_sum / max(seen, 1)
        end_avg = end_sum / max(seen, 1)
        junc_avg = junc_sum / max(seen, 1)
        total_avg = total_sum / max(seen, 1)

        row = {
            "epoch": epoch,
            "unsup_weight": wu,
            "skel_weight": args.skel_weight,
            "node_weight": args.node_weight,
            "train_total_loss": total_avg,
            "train_sup_loss": sup_avg,
            "train_cons_loss": con_avg,
            "train_skeleton_loss": skel_avg,
            "train_soft_cldice": scl_avg,
            "train_node_loss": node_avg,
            "train_endpoint_score": end_avg,
            "train_junction_score": junc_avg,
            "val_loss": val["loss"],
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_cldice": val["cldice"],
            "val_accuracy": val["accuracy"],
        }
        append_csv(history_path, row)

        state = {
            "epoch": epoch,
            "student_model": student.state_dict(),
            "teacher_model": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_dice": best_validation_dice,
            "global_step": global_step,
            "config": vars(args),
        }
        torch.save(state, checkpoint_dir / "last.pt")

        improved = val["dice"] > best_validation_dice
        if improved:
            best_validation_dice = val["dice"]
            state["best_val_dice"] = best_validation_dice
            torch.save(state, checkpoint_dir / "best.pt")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"sup {sup_avg:.4f} | con {con_avg:.5f} | "
            f"skel {skel_avg:.4f} | node {node_avg:.4f} | "
            f"E {end_avg:.4f} | J {junc_avg:.4f} | s-clDice {scl_avg:.4f} | "
            f"wu {wu:.3f} | Dice {val['dice']:.4f} | IoU {val['iou']:.4f} | "
            f"P {val['precision']:.4f} | R {val['recall']:.4f} | clDice {val['cldice']:.4f}"
            + ("  <-- BEST" if improved else "")
        )

    print("\nStructural-node training finished.")
    print("Best teacher validation Dice:", f"{best_validation_dice:.6f}")
    print("Best checkpoint:", checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
