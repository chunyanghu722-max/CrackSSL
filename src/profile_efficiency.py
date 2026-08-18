from pathlib import Path
import argparse
import json
import time

import torch

from baseline_utils import build_model


def load_model(checkpoint_path, model_kind, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = build_model(
        pretrained=False,
    ).to(device)

    if model_kind == "teacher":
        candidate_keys = [
            "teacher_model",
            "teacher",
            "model",
            "state_dict",
        ]
    else:
        candidate_keys = [
            "student_model",
            "student",
            "model",
            "state_dict",
        ]

    state_dict = None
    used_key = None

    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                used_key = key
                break

    if state_dict is None:
        if isinstance(checkpoint, dict) and all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        ):
            state_dict = checkpoint
            used_key = "<root>"
        else:
            raise KeyError(
                "Could not locate model state_dict."
            )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model, used_key


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def try_flops(
    model,
    dummy,
):
    # Preferred: fvcore
    try:
        from fvcore.nn import (
            FlopCountAnalysis,
        )

        analysis = FlopCountAnalysis(
            model,
            dummy,
        )

        total = float(
            analysis.total()
        )

        return total, "fvcore"
    except Exception as fvcore_error:
        fvcore_message = str(
            fvcore_error
        )

    # Fallback: THOP
    try:
        from thop import profile

        macs, _ = profile(
            model,
            inputs=(dummy,),
            verbose=False,
        )

        # Convention used here:
        # 1 MAC ~= 2 FLOPs.
        total = float(
            macs
        ) * 2.0

        return total, "thop (2 FLOPs/MAC)"
    except Exception as thop_error:
        thop_message = str(
            thop_error
        )

    return None, (
        "Unavailable. Install fvcore with: pip install fvcore "
        f"| fvcore error: {fvcore_message} "
        f"| thop error: {thop_message}"
    )


@torch.no_grad()
def measure_speed(
    model,
    dummy,
    device,
    warmup,
    runs,
):
    # Warm-up
    for _ in range(
        int(warmup)
    ):
        _ = model(
            dummy
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

        starter = torch.cuda.Event(
            enable_timing=True
        )

        ender = torch.cuda.Event(
            enable_timing=True
        )

        times_ms = []

        for _ in range(
            int(runs)
        ):
            starter.record()
            _ = model(
                dummy
            )
            ender.record()

            torch.cuda.synchronize()

            times_ms.append(
                float(
                    starter.elapsed_time(
                        ender
                    )
                )
            )
    else:
        times_ms = []

        for _ in range(
            int(runs)
        ):
            start = time.perf_counter()
            _ = model(
                dummy
            )
            end = time.perf_counter()

            times_ms.append(
                (
                    end - start
                )
                * 1000.0
            )

    tensor = torch.tensor(
        times_ms,
        dtype=torch.float64,
    )

    mean_ms = float(
        tensor.mean()
    )

    std_ms = float(
        tensor.std(
            unbiased=True
        )
        if tensor.numel() > 1
        else 0.0
    )

    batch_size = int(
        dummy.shape[0]
    )

    fps = (
        1000.0
        * batch_size
        / mean_ms
    )

    return (
        mean_ms,
        std_ms,
        fps,
    )


def human_number(value):
    if value is None:
        return "N/A"

    if value >= 1e9:
        return (
            f"{value / 1e9:.3f} G"
        )

    if value >= 1e6:
        return (
            f"{value / 1e6:.3f} M"
        )

    if value >= 1e3:
        return (
            f"{value / 1e3:.3f} K"
        )

    return f"{value:.0f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=[
            "teacher",
            "student",
        ],
        default="teacher",
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
        default=1,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--output",
        default="results/efficiency_full.json",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, checkpoint_key = (
        load_model(
            args.checkpoint,
            args.model,
            device,
        )
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    dummy = torch.randn(
        int(args.batch_size),
        3,
        int(args.height),
        int(args.width),
        device=device,
    )

    with torch.no_grad():
        flops, flops_backend = (
            try_flops(
                model,
                dummy,
            )
        )

    if device.type == "cuda":
        gpu_name = (
            torch.cuda.get_device_name(
                0
            )
        )
    else:
        gpu_name = "CPU"

    (
        mean_ms,
        std_ms,
        fps,
    ) = measure_speed(
        model,
        dummy,
        device,
        args.warmup,
        args.runs,
    )

    print("=" * 78)
    print(
        "MODEL EFFICIENCY PROFILE"
    )
    print("=" * 78)
    print(
        "Checkpoint      :",
        args.checkpoint,
    )
    print(
        "Checkpoint key  :",
        checkpoint_key,
    )
    print(
        "Device          :",
        device,
    )
    print(
        "Hardware        :",
        gpu_name,
    )
    print(
        "Input size      :",
        f"{args.batch_size} x 3 x {args.height} x {args.width}",
    )
    print(
        "Parameters      :",
        human_number(
            float(
                total_params
            )
        ),
        f"({total_params:,})",
    )
    print(
        "Trainable params:",
        human_number(
            float(
                trainable_params
            )
        ),
        f"({trainable_params:,})",
    )

    if flops is None:
        print(
            "FLOPs           : N/A"
        )
        print(
            "FLOPs backend   :",
            flops_backend,
        )
    else:
        print(
            "FLOPs           :",
            human_number(
                flops
            ),
        )
        print(
            "FLOPs backend   :",
            flops_backend,
        )

    print(
        "Latency         :",
        f"{mean_ms:.3f} +/- {std_ms:.3f} ms/batch",
    )
    print(
        "Throughput      :",
        f"{fps:.3f} images/s",
    )
    print(
        "Warm-up / Runs  :",
        f"{args.warmup} / {args.runs}",
    )
    print("=" * 78)

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_key": checkpoint_key,
        "model": args.model,
        "device": str(device),
        "hardware": gpu_name,
        "input": {
            "batch_size": int(
                args.batch_size
            ),
            "channels": 3,
            "height": int(
                args.height
            ),
            "width": int(
                args.width
            ),
        },
        "parameters": {
            "total": int(
                total_params
            ),
            "trainable": int(
                trainable_params
            ),
        },
        "flops": (
            None
            if flops is None
            else float(
                flops
            )
        ),
        "flops_backend": flops_backend,
        "latency_ms_per_batch": (
            mean_ms
        ),
        "latency_std_ms": (
            std_ms
        ),
        "fps_images_per_second": (
            fps
        ),
        "warmup": int(
            args.warmup
        ),
        "runs": int(
            args.runs
        ),
        "note": (
            "Skeleton consistency and contrastive projection head are "
            "training-only. Inference uses the same segmentation network."
        ),
    }

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
