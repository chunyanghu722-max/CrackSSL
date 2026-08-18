from pathlib import Path
import argparse
import json
import shutil
from collections import Counter

import numpy as np
from PIL import Image
import scipy.io as sio


def load_segmentation(mat_path):
    data = sio.loadmat(
        str(mat_path),
        squeeze_me=True,
        struct_as_record=False,
    )
    gt = data["groundTruth"]
    seg = np.asarray(gt.Segmentation)
    if seg.ndim != 2:
        raise ValueError(f"{mat_path}: unexpected segmentation shape {seg.shape}")
    return seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="datasets/CFD")
    parser.add_argument("--output-root", default="datasets/CFD_prepared")
    parser.add_argument("--keep-042", action="store_true")
    args = parser.parse_args()

    src = Path(args.source_root)
    img_root = src / "image"
    gt_root = src / "groundTruth"
    out = Path(args.output_root)
    test = out / "testcrop"

    if test.exists():
        shutil.rmtree(test)
    test.mkdir(parents=True, exist_ok=True)

    excluded = set()
    if not args.keep_042:
        excluded.add("042")

    label_counter = Counter()
    valid_ids = []
    empty_masks = []

    for i in range(1, 119):
        stem = f"{i:03d}"
        if stem in excluded:
            continue

        image_path = img_root / f"{stem}.jpg"
        mat_path = gt_root / f"{stem}.mat"

        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mat_path.exists():
            raise FileNotFoundError(mat_path)

        seg = load_segmentation(mat_path)
        labels, counts = np.unique(seg, return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            label_counter[int(label)] += int(count)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            w, h = image.size
            if seg.shape != (h, w):
                raise ValueError(
                    f"{stem}: image size {(w, h)} vs segmentation {seg.shape}"
                )
            image.save(test / f"{stem}.jpg", quality=95)

        # CFD convention: label 1 = surface, label 2 = crack.
        mask = (seg == 2).astype(np.uint8) * 255
        if int((mask > 0).sum()) == 0:
            empty_masks.append(stem)

        Image.fromarray(mask, mode="L").save(test / f"{stem}.png")
        valid_ids.append(stem)

    jpg_stems = {p.stem for p in test.glob("*.jpg")}
    png_stems = {p.stem for p in test.glob("*.png")}
    if jpg_stems != png_stems:
        raise RuntimeError("JPG/PNG pairing mismatch")

    report = {
        "protocol": "CFD zero-shot external test",
        "source_root": str(src),
        "output_root": str(out),
        "num_test_images": len(valid_ids),
        "excluded_ids": sorted(excluded),
        "ids": valid_ids,
        "segmentation_label_pixel_counts": {
            str(k): int(v) for k, v in sorted(label_counter.items())
        },
        "empty_crack_masks": empty_masks,
    }

    with open(out / "prepare_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("CFD PREPARATION COMPLETE")
    print("=" * 72)
    print("Test images:", len(valid_ids))
    print("Excluded:", sorted(excluded))
    print("Label pixel counts:", dict(sorted(label_counter.items())))
    print("Empty crack masks:", empty_masks)
    print("Pairing check: OK")
    print("Prepared folder:", test)


if __name__ == "__main__":
    main()
