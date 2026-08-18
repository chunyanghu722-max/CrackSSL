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
from topology_utils import (
    soft_cldice_consistency,
    distance_field_consistency,
)


def rampup(epoch, length):
    if length <= 0:
        return 1.0

    current = max(
        0.0,
        min(
            float(epoch),
            float(length),
        ),
    )

    phase = (
        1.0
        - current
        / float(length)
    )

    return math.exp(
        -5.0 * phase * phase
    )


def make_replacement_loader(
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


def make_eval_loader(
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

    total_loss = 0.0
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

        total_loss += (
            loss.item()
            * batch_size
        )

        seen += batch_size

        meter.update(
            logits.float(),
            masks,
        )

    metrics = meter.compute()

    metrics[
        "loss"
    ] = (
        total_loss
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

    parser.add_argument(
        "--skel-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--dist-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--skel-iters",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--dist-iters",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--topo-downsample",
        type=int,
        default=2,
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
            "--skel-weight must be >= 0."
        )

    if args.dist_weight < 0:
        raise ValueError(
            "--dist-weight must be >= 0."
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
        make_replacement_loader(
            labeled_dataset,
            args.batch_size,
            args.workers,
            args.epoch_samples,
            args.seed,
        )
    )

    unlabeled_loader = (
        make_replacement_loader(
            unlabeled_dataset,
            args.batch_size,
            args.workers,
            args.epoch_samples,
            args.seed + 1,
        )
    )

    validation_loader = (
        make_eval_loader(
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

    print("=" * 100)
    print(
        "CRACK500 EMA + SKELETON + DISTANCE-FIELD TOPOLOGY"
    )
    print("=" * 100)

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
        "Distance weight     :",
        args.dist_weight,
    )

    print(
        "Skeleton iterations :",
        args.skel_iters,
    )

    print(
        "Distance iterations :",
        args.dist_iters,
    )

    print(
        "Topology downsample :",
        args.topo_downsample,
    )

    print(
        "Ramp-up epochs      :",
        args.rampup_epochs,
    )

    print("=" * 100)

    student = build_model(
        pretrained=True,
    ).to(device)

    teacher = copy.deepcopy(
        student
    ).to(device)

    for parameter in teacher.parameters():
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

        unsupervised_weight = (
            args.max_unsup_weight
            * rampup(
                epoch - 1,
                args.rampup_epochs,
            )
        )

        supervised_sum = 0.0
        consistency_sum = 0.0
        skeleton_sum = 0.0
        soft_cldice_sum = 0.0
        distance_sum = 0.0
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

            if args.skel_weight > 0:
                (
                    skeleton_loss,
                    soft_cldice_score,
                ) = (
                    soft_cldice_consistency(
                        student_prob,
                        teacher_prob,
                        iterations=args.skel_iters,
                        downsample=args.topo_downsample,
                    )
                )
            else:
                skeleton_loss = (
                    student_prob.sum()
                    * 0.0
                )

                soft_cldice_score = (
                    torch.ones(
                        (),
                        device=device,
                    )
                )

            if args.dist_weight > 0:
                distance_loss = (
                    distance_field_consistency(
                        student_prob,
                        teacher_prob,
                        iterations=args.dist_iters,
                        downsample=args.topo_downsample,
                    )
                )
            else:
                distance_loss = (
                    student_prob.sum()
                    * 0.0
                )

            unsupervised_loss = (
                consistency_loss
                + args.skel_weight
                * skeleton_loss
                + args.dist_weight
                * distance_loss
            )

            total_loss = (
                supervised_loss
                + unsupervised_weight
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

            supervised_sum += (
                supervised_loss.item()
                * batch_size
            )

            consistency_sum += (
                consistency_loss.item()
                * batch_size
            )

            skeleton_sum += (
                skeleton_loss.item()
                * batch_size
            )

            soft_cldice_sum += (
                soft_cldice_score.item()
                * batch_size
            )

            distance_sum += (
                distance_loss.item()
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
                dist=(
                    f"{distance_loss.item():.4f}"
                ),
            )

        scheduler.step()

        validation = validate(
            teacher,
            validation_loader,
            device,
        )

        supervised_avg = (
            supervised_sum
            / max(seen, 1)
        )

        consistency_avg = (
            consistency_sum
            / max(seen, 1)
        )

        skeleton_avg = (
            skeleton_sum
            / max(seen, 1)
        )

        soft_cldice_avg = (
            soft_cldice_sum
            / max(seen, 1)
        )

        distance_avg = (
            distance_sum
            / max(seen, 1)
        )

        total_avg = (
            total_sum
            / max(seen, 1)
        )

        row = {
            "epoch": epoch,
            "unsup_weight": unsupervised_weight,
            "skel_weight": args.skel_weight,
            "dist_weight": args.dist_weight,

            "train_total_loss": total_avg,
            "train_sup_loss": supervised_avg,
            "train_cons_loss": consistency_avg,
            "train_skeleton_loss": skeleton_avg,
            "train_soft_cldice": soft_cldice_avg,
            "train_distance_loss": distance_avg,

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
            f"sup {supervised_avg:.4f} | "
            f"con {consistency_avg:.5f} | "
            f"skel {skeleton_avg:.4f} | "
            f"dist {distance_avg:.4f} | "
            f"s-clDice {soft_cldice_avg:.4f} | "
            f"wu {unsupervised_weight:.3f} | "
            f"Dice {validation['dice']:.4f} | "
            f"IoU {validation['iou']:.4f} | "
            f"P {validation['precision']:.4f} | "
            f"R {validation['recall']:.4f} | "
            f"clDice {validation['cldice']:.4f}"
            f"{marker}"
        )

    print(
        "\nFull topology training finished."
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
