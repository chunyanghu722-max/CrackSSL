from pathlib import Path
import argparse
import csv
import random

import cv2
import numpy as np

PCTS = [1, 5, 10, 20]
DEFAULT_SEEDS = [42, 3407, 2027]


def foreground_ratio(mask_path: Path) -> float:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    return float((mask > 127).mean())


def build_balanced_order(samples, seed: int, n_bins: int = 10):
    rng = random.Random(seed)
    sorted_samples = sorted(samples, key=lambda x: x["fg_ratio"])
    bins = [list(x) for x in np.array_split(sorted_samples, n_bins)]
    for b in bins:
        rng.shuffle(b)

    order = []
    max_len = max(len(b) for b in bins)
    bin_indices = list(range(len(bins)))

    for i in range(max_len):
        shuffled_bins = bin_indices[:]
        rng.shuffle(shuffled_bins)
        for bidx in shuffled_bins:
            if i < len(bins[bidx]):
                order.append(bins[bidx][i])

    assert len(order) == len(samples)
    assert len({x["stem"] for x in order}) == len(samples)
    return order


def write_list(path: Path, stems):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for stem in stems:
            f.write(stem + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="datasets/Crack500")
    parser.add_argument("--output-root", default="splits/Crack500")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    train_dir = Path(args.data_root) / "traincrop"
    out_root = Path(args.output_root)

    jpgs = sorted(train_dir.glob("*.jpg"))
    if not jpgs:
        raise RuntimeError(f"No JPG images found: {train_dir}")

    samples = []
    print(f"Reading {len(jpgs)} CRACK500 training masks...")
    for i, img_path in enumerate(jpgs, 1):
        mask_path = img_path.with_suffix(".png")
        if not mask_path.exists():
            raise RuntimeError(f"Missing mask: {mask_path}")

        samples.append({
            "stem": img_path.stem,
            "fg_ratio": foreground_ratio(mask_path),
        })

        if i % 250 == 0 or i == len(jpgs):
            print(f"  {i}/{len(jpgs)}")

    n_total = len(samples)
    all_ratio = np.array([x["fg_ratio"] for x in samples], dtype=np.float64)

    print("\nDataset statistics")
    print(f"Total train samples : {n_total}")
    print(f"Mean crack ratio    : {all_ratio.mean():.6f}")
    print(f"Median crack ratio  : {np.median(all_ratio):.6f}")
    print(f"Min / Max           : {all_ratio.min():.6f} / {all_ratio.max():.6f}")

    summary_rows = []

    for seed in args.seeds:
        seed_dir = out_root / f"seed{seed}"
        order = build_balanced_order(samples, seed=seed, n_bins=args.bins)

        all_stems = [x["stem"] for x in order]
        ratio_map = {x["stem"]: x["fg_ratio"] for x in samples}
        write_list(seed_dir / "all_train.txt", all_stems)

        print(f"\nSeed {seed}")
        previous_labeled = set()

        for pct in PCTS:
            n_labeled = int(round(n_total * pct / 100.0))
            labeled = all_stems[:n_labeled]
            unlabeled = all_stems[n_labeled:]

            labeled_set = set(labeled)
            if previous_labeled and not previous_labeled.issubset(labeled_set):
                raise RuntimeError("Nested split property failed.")
            previous_labeled = labeled_set

            write_list(seed_dir / f"labeled_{pct}pct.txt", labeled)
            write_list(seed_dir / f"unlabeled_{pct}pct.txt", unlabeled)

            labeled_ratios = np.array([ratio_map[s] for s in labeled], dtype=np.float64)
            unlabeled_ratios = np.array([ratio_map[s] for s in unlabeled], dtype=np.float64)

            summary_rows.append({
                "seed": seed,
                "pct": pct,
                "n_total": n_total,
                "n_labeled": len(labeled),
                "n_unlabeled": len(unlabeled),
                "dataset_mean_fg": float(all_ratio.mean()),
                "labeled_mean_fg": float(labeled_ratios.mean()),
                "labeled_median_fg": float(np.median(labeled_ratios)),
                "unlabeled_mean_fg": float(unlabeled_ratios.mean()) if len(unlabeled_ratios) else 0.0,
            })

            print(
                f"  {pct:>2}%: labeled={len(labeled):4d}, "
                f"unlabeled={len(unlabeled):4d}, "
                f"labeled crack ratio={labeled_ratios.mean():.5f}"
            )

    out_root.mkdir(parents=True, exist_ok=True)
    stats_path = out_root / "split_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 72)
    print("SSL SPLIT GENERATION: OK")
    print(f"Saved to: {out_root}")
    print(f"Statistics: {stats_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
