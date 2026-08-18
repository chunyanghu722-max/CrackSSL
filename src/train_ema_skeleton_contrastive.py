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
from contrastive_utils import (
    ContrastiveProjector,
    skeleton_guided_hard_negative_contrastive,
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
    generator.manual_seed(
        seed
    )

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


def forward_with_decoder_features(
    model,
    x,
):
    """
    Compatible with common segmentation_models_pytorch versions.
    """
    features = model.encoder(
        x
    )

    try:
        decoder_output = (
            model.decoder(
                features
            )
        )
    except TypeError:
        decoder_output = (
            model.decoder(
                *features
            )
        )

    logits = (
        model.segmentation_head(
            decoder_output
        )
    )

    return (
        logits,
        decoder_output,
    )


@torch.no_grad()
def infer_decoder_channels(
    model,
    height,
    width,
    device,
):
    was_training = (
        model.training
    )

    model.eval()

    dummy = torch.zeros(
        1,
        3,
        int(height),
        int(width),
        device=device,
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
        _, feature = (
            forward_with_decoder_features(
                model,
                dummy,
            )
        )

    channels = int(
        feature.shape[1]
    )

    spatial = (
        int(feature.shape[2]),
        int(feature.shape[3]),
    )

    del dummy
    del feature

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.train(
        was_training
    )

    return channels, spatial


@torch.no_grad()
def update_ema_module(
    teacher_module,
    student_module,
    decay,
    global_step,
):
    alpha = min(
        1.0
        - 1.0
        / float(
            global_step + 1
        ),
        float(
            decay
        ),
    )

    for (
        teacher_parameter,
        student_parameter,
    ) in zip(
        teacher_module.parameters(),
        student_module.parameters(),
    ):
        teacher_parameter.data.mul_(
            alpha
        ).add_(
            student_parameter.data,
            alpha=1.0 - alpha,
        )

    for (
        teacher_buffer,
        student_buffer,
    ) in zip(
        teacher_module.buffers(),
        student_module.buffers(),
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
        / max(
            seen,
            1,
        )
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

    # Successful Step 11 term is frozen.
    parser.add_argument(
        "--topo-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--topo-iters",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--topo-downsample",
        type=int,
        default=2,
    )

    # New Step 15 representation-learning term.
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--projection-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--contrast-pos-threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--contrast-neg-threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--hard-negative-radius",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-positive-samples",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--max-negative-samples",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--contrast-skel-iters",
        type=int,
        default=6,
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

    if args.topo_weight < 0:
        raise ValueError(
            "--topo-weight must be >= 0."
        )

    if args.contrast_weight < 0:
        raise ValueError(
            "--contrast-weight must be >= 0."
        )

    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be > 0."
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

    student = build_model(
        pretrained=True,
    ).to(
        device
    )

    teacher = copy.deepcopy(
        student
    ).to(
        device
    )

    for parameter in (
        teacher.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    teacher.eval()

    (
        decoder_channels,
        decoder_spatial,
    ) = infer_decoder_channels(
        student,
        args.height,
        args.width,
        device,
    )

    student_projector = (
        ContrastiveProjector(
            decoder_channels,
            projection_dim=(
                args.projection_dim
            ),
        ).to(
            device
        )
    )

    teacher_projector = (
        copy.deepcopy(
            student_projector
        ).to(
            device
        )
    )

    for parameter in (
        teacher_projector.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    teacher_projector.eval()

    print("=" * 118)
    print(
        "CRACK500 EMA + SKELETON + SKELETON-GUIDED HARD-NEGATIVE CONTRASTIVE LEARNING"
    )
    print("=" * 118)

    print(
        "Device                 :",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU                    :",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Unique labeled         :",
        len(
            labeled_dataset
        ),
    )

    print(
        "Unique unlabeled       :",
        len(
            unlabeled_dataset
        ),
    )

    print(
        "Updates/epoch          :",
        len(
            labeled_loader
        ),
    )

    print(
        "EMA decay              :",
        args.ema_decay,
    )

    print(
        "Skeleton weight        :",
        args.topo_weight,
    )

    print(
        "Contrastive weight     :",
        args.contrast_weight,
    )

    print(
        "Decoder feature        :",
        decoder_channels,
        "channels @",
        decoder_spatial,
    )

    print(
        "Projection dim         :",
        args.projection_dim,
    )

    print(
        "Temperature            :",
        args.temperature,
    )

    print(
        "Positive threshold     :",
        args.contrast_pos_threshold,
    )

    print(
        "Negative threshold     :",
        args.contrast_neg_threshold,
    )

    print(
        "Hard-negative radius   :",
        args.hard_negative_radius,
        "feature pixels",
    )

    print(
        "Positive samples max   :",
        args.max_positive_samples,
    )

    print(
        "Hard-negative max      :",
        args.max_negative_samples,
    )

    print(
        "Ramp-up epochs         :",
        args.rampup_epochs,
    )

    print("=" * 118)

    optimizer = (
        torch.optim.AdamW(
            list(
                student.parameters()
            )
            + list(
                student_projector.parameters()
            ),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
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
        student_projector.train()

        teacher.eval()
        teacher_projector.eval()

        unsupervised_weight = (
            args.max_unsup_weight
            * rampup(
                epoch - 1,
                args.rampup_epochs,
            )
        )

        supervised_sum = 0.0
        consistency_sum = 0.0
        topology_sum = 0.0
        topology_score_sum = 0.0
        contrast_sum = 0.0

        valid_contrast_sum = 0.0
        positive_sample_sum = 0.0
        negative_sample_sum = 0.0
        positive_similarity_sum = 0.0
        hard_negative_similarity_sum = 0.0

        total_sum = 0.0
        seen = 0

        progress = tqdm(
            zip(
                labeled_loader,
                unlabeled_loader,
            ),
            total=min(
                len(
                    labeled_loader
                ),
                len(
                    unlabeled_loader
                ),
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

            # ------------------------------------------
            # EMA Teacher: weak view
            # ------------------------------------------
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
                    (
                        teacher_logits,
                        teacher_decoder_feature,
                    ) = (
                        forward_with_decoder_features(
                            teacher,
                            unlabeled_weak,
                        )
                    )

                    teacher_projection = (
                        teacher_projector(
                            teacher_decoder_feature
                        )
                    )

                teacher_probability = (
                    torch.sigmoid(
                        teacher_logits.float()
                    )
                )

            # ------------------------------------------
            # Student supervised + unlabeled strong view
            # ------------------------------------------
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

                (
                    student_unlabeled_logits,
                    student_decoder_feature,
                ) = (
                    forward_with_decoder_features(
                        student,
                        unlabeled_strong,
                    )
                )

                student_projection = (
                    student_projector(
                        student_decoder_feature
                    )
                )

            student_probability = (
                torch.sigmoid(
                    student_unlabeled_logits.float()
                )
            )

            # ------------------------------------------
            # Pixel consistency
            # ------------------------------------------
            consistency_loss = (
                F.mse_loss(
                    student_probability,
                    teacher_probability,
                )
            )

            # ------------------------------------------
            # Successful centerline topology consistency
            # ------------------------------------------
            (
                topology_loss,
                topology_score,
            ) = (
                soft_cldice_consistency(
                    student_probability,
                    teacher_probability,
                    iterations=args.topo_iters,
                    downsample=args.topo_downsample,
                )
            )

            # ------------------------------------------
            # New feature-space contrastive branch
            # ------------------------------------------
            (
                contrastive_loss,
                contrastive_stats,
            ) = (
                skeleton_guided_hard_negative_contrastive(
                    student_projection,
                    teacher_projection,
                    teacher_probability,
                    temperature=args.temperature,
                    positive_threshold=(
                        args.contrast_pos_threshold
                    ),
                    negative_threshold=(
                        args.contrast_neg_threshold
                    ),
                    hard_negative_radius=(
                        args.hard_negative_radius
                    ),
                    max_positive_samples=(
                        args.max_positive_samples
                    ),
                    max_negative_samples=(
                        args.max_negative_samples
                    ),
                    skeleton_iterations=(
                        args.contrast_skel_iters
                    ),
                )
            )

            unsupervised_loss = (
                consistency_loss
                + args.topo_weight
                * topology_loss
                + args.contrast_weight
                * contrastive_loss
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

            update_ema_module(
                teacher,
                student,
                args.ema_decay,
                global_step,
            )

            update_ema_module(
                teacher_projector,
                student_projector,
                args.ema_decay,
                global_step,
            )

            batch_size = (
                labeled_images.size(
                    0
                )
            )

            supervised_sum += (
                supervised_loss.item()
                * batch_size
            )

            consistency_sum += (
                consistency_loss.item()
                * batch_size
            )

            topology_sum += (
                topology_loss.item()
                * batch_size
            )

            topology_score_sum += (
                topology_score.item()
                * batch_size
            )

            contrast_sum += (
                contrastive_loss.item()
                * batch_size
            )

            valid_contrast_sum += (
                contrastive_stats[
                    "valid_images"
                ]
            )

            positive_sample_sum += (
                contrastive_stats[
                    "positive_samples"
                ]
                * batch_size
            )

            negative_sample_sum += (
                contrastive_stats[
                    "negative_samples"
                ]
                * batch_size
            )

            positive_similarity_sum += (
                contrastive_stats[
                    "positive_similarity"
                ]
                * batch_size
            )

            hard_negative_similarity_sum += (
                contrastive_stats[
                    "hard_negative_similarity"
                ]
                * batch_size
            )

            total_sum += (
                total_loss.item()
                * batch_size
            )

            seen += (
                batch_size
            )

            progress.set_postfix(
                sup=(
                    f"{supervised_loss.item():.4f}"
                ),
                topo=(
                    f"{topology_loss.item():.4f}"
                ),
                ctr=(
                    f"{contrastive_loss.item():.4f}"
                ),
                vImg=(
                    f"{contrastive_stats['valid_images']:.0f}"
                ),
            )

        scheduler.step()

        validation = (
            validate(
                teacher,
                validation_loader,
                device,
            )
        )

        def average(value):
            return (
                value
                / max(
                    seen,
                    1,
                )
            )

        row = {
            "epoch": epoch,
            "unsup_weight": (
                unsupervised_weight
            ),
            "topo_weight": (
                args.topo_weight
            ),
            "contrast_weight": (
                args.contrast_weight
            ),

            "train_total_loss": (
                average(
                    total_sum
                )
            ),
            "train_sup_loss": (
                average(
                    supervised_sum
                )
            ),
            "train_cons_loss": (
                average(
                    consistency_sum
                )
            ),
            "train_topology_loss": (
                average(
                    topology_sum
                )
            ),
            "train_soft_cldice": (
                average(
                    topology_score_sum
                )
            ),
            "train_contrastive_loss": (
                average(
                    contrast_sum
                )
            ),

            "train_valid_contrast_images_per_batch": (
                valid_contrast_sum
                / max(
                    len(
                        labeled_loader
                    ),
                    1,
                )
            ),
            "train_positive_samples": (
                average(
                    positive_sample_sum
                )
            ),
            "train_negative_samples": (
                average(
                    negative_sample_sum
                )
            ),
            "train_positive_similarity": (
                average(
                    positive_similarity_sum
                )
            ),
            "train_hard_negative_similarity": (
                average(
                    hard_negative_similarity_sum
                )
            ),

            "val_loss": (
                validation[
                    "loss"
                ]
            ),
            "val_dice": (
                validation[
                    "dice"
                ]
            ),
            "val_iou": (
                validation[
                    "iou"
                ]
            ),
            "val_precision": (
                validation[
                    "precision"
                ]
            ),
            "val_recall": (
                validation[
                    "recall"
                ]
            ),
            "val_cldice": (
                validation[
                    "cldice"
                ]
            ),
            "val_accuracy": (
                validation[
                    "accuracy"
                ]
            ),
        }

        append_csv(
            history_path,
            row,
        )

        state = {
            "epoch": epoch,
            "student_model": (
                student.state_dict()
            ),
            "teacher_model": (
                teacher.state_dict()
            ),
            "student_projector": (
                student_projector.state_dict()
            ),
            "teacher_projector": (
                teacher_projector.state_dict()
            ),
            "optimizer": (
                optimizer.state_dict()
            ),
            "scheduler": (
                scheduler.state_dict()
            ),
            "best_val_dice": (
                best_validation_dice
            ),
            "global_step": (
                global_step
            ),
            "config": (
                vars(
                    args
                )
            ),
        }

        torch.save(
            state,
            checkpoint_dir
            / "last.pt",
        )

        improved = (
            validation[
                "dice"
            ]
            > best_validation_dice
        )

        if improved:
            best_validation_dice = (
                validation[
                    "dice"
                ]
            )

            state[
                "best_val_dice"
            ] = (
                best_validation_dice
            )

            torch.save(
                state,
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
            f"sup {average(supervised_sum):.4f} | "
            f"con {average(consistency_sum):.5f} | "
            f"topo {average(topology_sum):.4f} | "
            f"ctr {average(contrast_sum):.4f} | "
            f"pos {average(positive_sample_sum):.1f} | "
            f"neg {average(negative_sample_sum):.1f} | "
            f"sim+ {average(positive_similarity_sum):.3f} | "
            f"simHard {average(hard_negative_similarity_sum):.3f} | "
            f"wu {unsupervised_weight:.3f} | "
            f"Dice {validation['dice']:.4f} | "
            f"IoU {validation['iou']:.4f} | "
            f"P {validation['precision']:.4f} | "
            f"R {validation['recall']:.4f} | "
            f"clDice {validation['cldice']:.4f}"
            f"{marker}"
        )

    print(
        "\nStep 15 training finished."
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
