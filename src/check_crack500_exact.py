from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "datasets"

EXPECTED = {
    "traincrop": 1896,
    "valcrop": 348,
    "testcrop": 1124,
}

def check_split(split_dir: Path, expected: int):
    if not split_dir.is_dir():
        print(f"[ERROR] Missing folder: {split_dir}")
        return False

    jpgs = sorted(split_dir.glob("*.jpg"))
    pngs = sorted(split_dir.glob("*.png"))

    jpg_stems = {p.stem for p in jpgs}
    png_stems = {p.stem for p in pngs}

    missing_masks = sorted(jpg_stems - png_stems)
    missing_images = sorted(png_stems - jpg_stems)
    paired = jpg_stems & png_stems

    print(f"\n[{split_dir.name}]")
    print(f"JPG images : {len(jpgs)}")
    print(f"PNG masks  : {len(pngs)}")
    print(f"Pairs      : {len(paired)}")
    print(f"Expected   : {expected}")
    print(f"Pair names : {'OK' if not missing_masks and not missing_images else 'ERROR'}")
    print(f"Count      : {'OK' if len(paired) == expected else 'WARNING'}")

    if missing_masks:
        print("Images without masks (first 10):")
        for x in missing_masks[:10]:
            print("  ", x)
    if missing_images:
        print("Masks without images (first 10):")
        for x in missing_images[:10]:
            print("  ", x)

    return (
        len(jpgs) == len(pngs)
        and not missing_masks
        and not missing_images
        and len(paired) == expected
    )

def main():
    crack500 = DATA_ROOT / "Crack500"
    print("=" * 72)
    print("CRACK500 exact pairing checker")
    print("Dataset:", crack500)
    print("=" * 72)

    if not crack500.is_dir():
        print("[ERROR] datasets/Crack500 was not found.")
        sys.exit(1)

    results = []
    for split, expected in EXPECTED.items():
        results.append(check_split(crack500 / split, expected))

    print("\n" + "=" * 72)
    if all(results):
        print("CRACK500 CHECK: OK")
        print("All train/val/test images and masks are paired correctly.")
        sys.exit(0)
    else:
        print("CRACK500 CHECK: FAILED/WARNING")
        print("Do not start training until the messages above are resolved.")
        sys.exit(1)

if __name__ == "__main__":
    main()
