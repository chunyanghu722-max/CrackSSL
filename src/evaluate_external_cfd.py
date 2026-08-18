from pathlib import Path
import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from crack500_subset_data import Crack500SubsetDataset
from baseline_utils import (
    BinarySegmentationMeter,
    bce_dice_loss,
    build_model,
)


def load_checkpoint_model(checkpoint_path, model_kind, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False).to(device)

    keys = (
        ["teacher_model", "teacher", "model", "state_dict"]
        if model_kind == "teacher"
        else ["student_model", "student", "model", "state_dict"]
    )

    state_dict = None
    used_key = None

    if isinstance(checkpoint, dict):
        for key in keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                used_key = key
                break

    if state_dict is None:
        if isinstance(checkpoint, dict) and all(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            state_dict = checkpoint
            used_key = "<root>"
        else:
            raise KeyError(
                f"Could not find {model_kind} weights. "
                f"Keys={list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
            )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, used_key


@torch.no_grad()
def evaluate(model, loader, device):
    meter = BinarySegmentationMeter(
        threshold=0.5,
        compute_cldice=True,
    )
    loss_sum = 0.0
    seen = 0

    for batch in tqdm(loader, desc="CFD External Test"):
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
        loss_sum += float(loss.item()) * bs
        seen += bs
        meter.update(logits.float(), masks)

    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(seen, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--data-root", default="datasets/CFD_prepared")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Crack500SubsetDataset(
        args.data_root,
        split="testcrop",
        list_file=None,
        height=args.height,
        width=args.width,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=False,
    )

    model, used_key = load_checkpoint_model(
        args.checkpoint,
        args.model,
        device,
    )

    print("=" * 78)
    print("CROSS-DATASET ZERO-SHOT TEST: CRACK500 -> CFD")
    print("=" * 78)
    print("Checkpoint:", args.checkpoint)
    print("Model:", args.model)
    print("Checkpoint key:", used_key)
    print("CFD samples:", len(dataset))
    print("Device:", device)

    metrics = evaluate(model, loader, device)

    print("")
    print("=" * 78)
    print("CFD EXTERNAL TEST RESULTS")
    print("=" * 78)

    for key in [
        "dice",
        "f1",
        "iou",
        "precision",
        "recall",
        "cldice",
        "accuracy",
        "loss",
    ]:
        if key in metrics:
            print(f"{key:>12}: {metrics[key]:.6f}")

    checkpoint_path = Path(args.checkpoint)
    if args.output is None:
        output_path = (
            checkpoint_path.parent.parent
            / f"cfd_external_metrics_{args.model}.json"
        )
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "protocol": (
            "Zero-shot cross-dataset generalization: "
            "trained on Crack500, directly evaluated on CFD."
        ),
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "data_root": args.data_root,
        "num_samples": len(dataset),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
