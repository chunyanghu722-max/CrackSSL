from pathlib import Path
import argparse
import copy
import math

import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    RandomSampler,
)
from tqdm import tqdm

from crack500_subset_data import (
    Crack500SubsetDataset,
)
from ema_data import (
    Crack500UnlabeledPairDataset,
)
from baseline_utils import (
    BinarySegmentationMeter,
    append_csv,
    bce_dice_loss,
    build_model,
    save_json,
    set_seed,
)
from topology_utils import (
    soft_cldice_consistency,
)
from gap_topology_utils import (
    gap_repair_loss,
)


def rampup(epoch, length):
    if length <= 0:
        return 1.0

    x = max(
        0.0,
        min(
            float(epoch),
            float(length),
        ),
    )

    phase = (
        1.0
        - x / float(length)
    )

    return math.exp(
        -5.0
        * phase
        * phase
    )


def replacement_loader(
    dataset,
    batch_size,
    workers,
    epoch_samples,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

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
        persistent_workers=(
            workers > 0
        ),
        drop_last=False,
    )


def evaluation_loader(
    dataset,
    batch_size,
    workers,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(
            workers > 0
        ),
        drop_last=False,
    )


@torch.no_grad()
def update_teacher(
    teacher,
    student,
    decay,
    global_step,
):
    alpha = min(
        1.0
        - 1.0
        / float(global_step + 1),
        decay,
    )

    for teacher_param, student_param in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        teacher_param.data.mul_(
            alpha
        ).add_(
            student_param.data,
            alpha=1.0 - alpha,
        )

    for teacher_buffer, student_buffer in zip(
        teacher.buffers(),
        student.buffers(),
    ):
        teacher_buffer.copy_(
            student_buffer
        )


@torch.no_grad()
def validate(
    model,
    loader,
    device,
):
    model.eval()

    meter = BinarySegmentationMeter(
        threshold=0.5,
        compute_cldice=True,
    )

    loss_sum = 0.0
    seen = 0

    for batch in tqdm(
        loader,
        desc="Val",
        leave=False,
    ):
        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        masks = batch[
            "mask"
        ].to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.bfloat16
            ),
            enabled=(
                device.type == "cuda"
            ),
        ):
            logits = model(
                images
            )

            loss, _, _ = (
                bce_dice_loss(
                    logits,
                    masks,
                )
            )

        batch_size = (
            images.size(0)
        )

        loss_sum += (
            loss.item()
            * batch_size
        )

        seen += batch_size

        meter.update(
            logits.float(),
            masks,
        )

    metrics = (
        meter.compute()
    )

    metrics[
        "loss"
    ] = (
        loss_sum
        / max(seen, 1)
    )

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

    # Frozen successful Step 11 weight.
    parser.add_argument(
        "--skel-weight",
        type=float,
        default=0.05,
    )

    # New conservative gap repair term.
    parser.add_argument(
        "--gap-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--skel-iters",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--topo-downsample",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--gap-max-distance",
        type=float,
        default=12.0,
        help=(
            "Max endpoint-pair distance on downsampled topology map."
        ),
    )

    parser.add_argument(
        "--gap-min-alignment",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--gap-min-line-prob",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--gap-guard-weight",
        type=float,
        default=0.5,
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

    if args.skel_weight < 0:
        raise ValueError(
            "--skel-weight must be >= 0"
        )

    if args.gap_weight < 0:
        raise ValueError(
            "--gap-weight must be >= 0"
        )

    set_seed(
        args.seed
    )

    output_dir = Path(
        args.output
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        vars(args),
        output_dir
        / "config.json",
    )

    labeled_dataset = (
        Crack500SubsetDataset(
            args.data_root,
            split="traincrop",
            list_file=args.labeled_list,
            height=args.height,
            width=args.width,
            train=True,
        )
    )

    unlabeled_dataset = (
        Crack500UnlabeledPairDataset(
            args.data_root,
            args.unlabeled_list,
            args.height,
            args.width,
        )
    )

    validation_dataset = (
        Crack500SubsetDataset(
            args.data_root,
            split="valcrop",
            list_file=None,
            height=args.height,
            width=args.width,
            train=False,
        )
    )

    labeled_loader = (
        replacement_loader(
            labeled_dataset,
            args.batch_size,
            args.workers,
            args.epoch_samples,
            args.seed,
        )
    )

    unlabeled_loader = (
        replacement_loader(
            unlabeled_dataset,
            args.batch_size,
            args.workers,
            args.epoch_samples,
            args.seed + 1,
        )
    )

    validation_loader = (
        evaluation_loader(
            validation_dataset,
            args.batch_size,
            args.workers,
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 112)
    print(
        "CRACK500 EMA + SKELETON + GAP-AWARE BRIDGE/GUARD REPAIR"
    )
    print("=" * 112)

    print(
        "Device              :",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU                 :",
            torch.cuda.get_device_name(0),
        )

    print(
        "Unique labeled      :",
        len(labeled_dataset),
    )

    print(
        "Unique unlabeled    :",
        len(unlabeled_dataset),
    )

    print(
        "Updates/epoch       :",
        len(labeled_loader),
    )

    print(
        "EMA decay           :",
        args.ema_decay,
    )

    print(
        "Skeleton weight     :",
        args.skel_weight,
    )

    print(
        "Gap weight          :",
        args.gap_weight,
    )

    print(
        "Gap max distance    :",
        args.gap_max_distance,
        "(topology-map pixels)",
    )

    print(
        "Gap min alignment   :",
        args.gap_min_alignment,
    )

    print(
        "Gap min line prob   :",
        args.gap_min_line_prob,
    )

    print(
        "Guard weight        :",
        args.gap_guard_weight,
    )

    print(
        "Topology downsample :",
        args.topo_downsample,
    )

    print(
        "Ramp-up epochs      :",
        args.rampup_epochs,
    )

    print("=" * 112)

    student = build_model(
        pretrained=True,
    ).to(device)

    teacher = copy.deepcopy(
        student
    ).to(device)

    for parameter in (
        teacher.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    teacher.eval()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    best_validation_dice = -1.0
    global_step = 0

    history_path = (
        output_dir
        / "history.csv"
    )

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
        skel_sum = 0.0
        scldice_sum = 0.0
        gap_sum = 0.0
        bridge_sum = 0.0
        guard_sum = 0.0
        gap_count_sum = 0.0
        bridge_ratio_sum = 0.0
        total_sum = 0.0
        seen = 0

        progress = tqdm(
            zip(
                labeled_loader,
                unlabeled_loader,
            ),
            total=min(
                len(labeled_loader),
                len(unlabeled_loader),
            ),
            desc="Train",
            leave=False,
        )

        for (
            labeled_batch,
            unlabeled_batch,
        ) in progress:
            labeled_images = (
                labeled_batch[
                    "image"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            labeled_masks = (
                labeled_batch[
                    "mask"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            unlabeled_weak = (
                unlabeled_batch[
                    "weak"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            unlabeled_strong = (
                unlabeled_batch[
                    "strong"
                ].to(
                    device,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=(
                        torch.float16
                        if device.type == "cuda"
                        else torch.bfloat16
                    ),
                    enabled=(
                        device.type == "cuda"
                    ),
                ):
                    teacher_logits = (
                        teacher(
                            unlabeled_weak
                        )
                    )

                teacher_prob = (
                    torch.sigmoid(
                        teacher_logits.float()
                    )
                )

            with torch.amp.autocast(
                device_type=device.type,
                dtype=(
                    torch.float16
                    if device.type == "cuda"
                    else torch.bfloat16
                ),
                enabled=(
                    device.type == "cuda"
                ),
            ):
                labeled_logits = (
                    student(
                        labeled_images
                    )
                )

                supervised_loss, _, _ = (
                    bce_dice_loss(
                        labeled_logits,
                        labeled_masks,
                    )
                )

                student_unlabeled_logits = (
                    student(
                        unlabeled_strong
                    )
                )

            student_prob = (
                torch.sigmoid(
                    student_unlabeled_logits.float()
                )
            )

            consistency_loss = (
                F.mse_loss(
                    student_prob,
                    teacher_prob,
                )
            )

            (
                skeleton_loss,
                soft_cldice_score,
            ) = soft_cldice_consistency(
                student_prob,
                teacher_prob,
                iterations=args.skel_iters,
                downsample=args.topo_downsample,
            )

            (
                gap_loss,
                bridge_loss,
                guard_loss,
                mean_gap_count,
                bridge_ratio,
            ) = gap_repair_loss(
                student_prob,
                teacher_prob,
                threshold=0.5,
                downsample=args.topo_downsample,
                max_distance=args.gap_max_distance,
                min_alignment=args.gap_min_alignment,
                min_line_prob=args.gap_min_line_prob,
                guard_weight=args.gap_guard_weight,
            )

            unsupervised_loss = (
                consistency_loss
                + args.skel_weight
                * skeleton_loss
                + args.gap_weight
                * gap_loss
            )

            total_loss = (
                supervised_loss
                + unsup_weight
                * unsupervised_loss
            )

            if not torch.isfinite(
                total_loss
            ):
                raise RuntimeError(
                    "Non-finite loss detected."
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

            batch_size = (
                labeled_images.size(0)
            )

            sup_sum += (
                supervised_loss.item()
                * batch_size
            )

            con_sum += (
                consistency_loss.item()
                * batch_size
            )

            skel_sum += (
                skeleton_loss.item()
                * batch_size
            )

            scldice_sum += (
                soft_cldice_score.item()
                * batch_size
            )

            gap_sum += (
                gap_loss.item()
                * batch_size
            )

            bridge_sum += (
                bridge_loss.item()
                * batch_size
            )

            guard_sum += (
                guard_loss.item()
                * batch_size
            )

            gap_count_sum += (
                mean_gap_count
                * batch_size
            )

            bridge_ratio_sum += (
                bridge_ratio
                * batch_size
            )

            total_sum += (
                total_loss.item()
                * batch_size
            )

            seen += batch_size

            progress.set_postfix(
                sup=(
                    f"{supervised_loss.item():.4f}"
                ),
                con=(
                    f"{consistency_loss.item():.5f}"
                ),
                skel=(
                    f"{skeleton_loss.item():.4f}"
                ),
                gap=(
                    f"{gap_loss.item():.4f}"
                ),
                nGap=(
                    f"{mean_gap_count:.2f}"
                ),
            )

        scheduler.step()

        validation = validate(
            teacher,
            validation_loader,
            device,
        )

        def avg(value):
            return (
                value
                / max(seen, 1)
            )

        row = {
            "epoch": epoch,
            "unsup_weight": unsup_weight,
            "skel_weight": args.skel_weight,
            "gap_weight": args.gap_weight,

            "train_total_loss": avg(total_sum),
            "train_sup_loss": avg(sup_sum),
            "train_cons_loss": avg(con_sum),
            "train_skeleton_loss": avg(skel_sum),
            "train_soft_cldice": avg(scldice_sum),

            "train_gap_loss": avg(gap_sum),
            "train_bridge_loss": avg(bridge_sum),
            "train_guard_loss": avg(guard_sum),
            "train_mean_gap_count": avg(gap_count_sum),
            "train_bridge_ratio": avg(bridge_ratio_sum),

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

        checkpoint = {
            "epoch": epoch,
            "student_model": student.state_dict(),
            "teacher_model": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_dice": best_validation_dice,
            "global_step": global_step,
            "config": vars(args),
        }

        torch.save(
            checkpoint,
            checkpoint_dir
            / "last.pt",
        )

        improved = (
            validation["dice"]
            > best_validation_dice
        )

        if improved:
            best_validation_dice = (
                validation["dice"]
            )

            checkpoint[
                "best_val_dice"
            ] = (
                best_validation_dice
            )

            torch.save(
                checkpoint,
                checkpoint_dir
                / "best.pt",
            )

        marker = (
            "  <-- BEST"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"sup {avg(sup_sum):.4f} | "
            f"con {avg(con_sum):.5f} | "
            f"skel {avg(skel_sum):.4f} | "
            f"gap {avg(gap_sum):.4f} | "
            f"bridge {avg(bridge_sum):.4f} | "
            f"guard {avg(guard_sum):.4f} | "
            f"nGap {avg(gap_count_sum):.2f} | "
            f"bRatio {avg(bridge_ratio_sum):.6f} | "
            f"s-clDice {avg(scldice_sum):.4f} | "
            f"wu {unsup_weight:.3f} | "
            f"Dice {validation['dice']:.4f} | "
            f"IoU {validation['iou']:.4f} | "
            f"P {validation['precision']:.4f} | "
            f"R {validation['recall']:.4f} | "
            f"clDice {validation['cldice']:.4f}"
            f"{marker}"
        )

    print(
        "\nGap-aware topology training finished."
    )

    print(
        "Best teacher validation Dice:",
        f"{best_validation_dice:.6f}",
    )

    print(
        "Best checkpoint:",
        checkpoint_dir
        / "best.pt",
    )


if __name__ == "__main__":
    main()
