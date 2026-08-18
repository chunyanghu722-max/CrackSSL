from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "datasets"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def files_with_ext(folder, exts):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def stems(files):
    return {p.stem for p in files}


def status(ok):
    return "OK" if ok else "ERROR"


def check_deepcrack(path):
    print("\n[DeepCrack]")
    required = ["train_img", "train_lab", "test_img", "test_lab"]
    missing = [x for x in required if not (path / x).is_dir()]
    if missing:
        print(f"ERROR: missing folders: {', '.join(missing)}")
        return False

    tr_img = files_with_ext(path / "train_img", IMG_EXTS)
    tr_lab = files_with_ext(path / "train_lab", IMG_EXTS)
    te_img = files_with_ext(path / "test_img", IMG_EXTS)
    te_lab = files_with_ext(path / "test_lab", IMG_EXTS)

    tr_img_stems, tr_lab_stems = stems(tr_img), stems(tr_lab)
    te_img_stems, te_lab_stems = stems(te_img), stems(te_lab)
    tr_match = tr_img_stems == tr_lab_stems
    te_match = te_img_stems == te_lab_stems

    print(f"train_img: {len(tr_img)} | train_lab: {len(tr_lab)} | pair-by-name: {status(tr_match)}")
    print(f"test_img : {len(te_img)} | test_lab : {len(te_lab)} | pair-by-name: {status(te_match)}")

    if not tr_match:
        print("  unmatched train images:", sorted(tr_img_stems - tr_lab_stems)[:10])
        print("  unmatched train labels:", sorted(tr_lab_stems - tr_img_stems)[:10])
    if not te_match:
        print("  unmatched test images:", sorted(te_img_stems - te_lab_stems)[:10])
        print("  unmatched test labels:", sorted(te_lab_stems - te_img_stems)[:10])

    canonical_counts = (len(tr_img), len(tr_lab), len(te_img), len(te_lab)) == (300, 300, 237, 237)
    if canonical_counts:
        print("Canonical DeepCrack counts: OK (300 train + 237 test)")
    else:
        print("NOTE: counts differ from the canonical DeepCrack release (300/300/237/237).")

    return tr_match and te_match and all([tr_img, tr_lab, te_img, te_lab])


def check_cfd(path):
    print("\n[CFD / CrackForest]")
    required = ["image", "groundTruth", "seg"]
    missing = [x for x in required if not (path / x).is_dir()]
    if missing:
        print(f"ERROR: missing folders: {', '.join(missing)}")
        return False

    images = files_with_ext(path / "image", IMG_EXTS)
    mats = files_with_ext(path / "groundTruth", {".mat"})
    segs = files_with_ext(path / "seg", {".seg"})
    i, m, s = stems(images), stems(mats), stems(segs)

    annotated_stems = m & s
    missing_image_for_gt = annotated_stems - i
    gt_without_seg = m - s
    seg_without_gt = s - m
    extra_unannotated_images = i - annotated_stems

    print(f"image: {len(images)} | groundTruth (.mat): {len(mats)} | seg: {len(segs)}")
    print(f"annotated samples with image+MAT+SEG: {len(annotated_stems & i)}")
    print(f"extra images without both annotations: {len(extra_unannotated_images)}")

    ok = not missing_image_for_gt and not gt_without_seg and not seg_without_gt and len(annotated_stems & i) > 0
    print(f"annotation pairing: {status(ok)}")
    if missing_image_for_gt:
        print("  annotations without image:", sorted(missing_image_for_gt)[:10])
    if gt_without_seg:
        print("  MAT without SEG:", sorted(gt_without_seg)[:10])
    if seg_without_gt:
        print("  SEG without MAT:", sorted(seg_without_gt)[:10])

    if len(annotated_stems & i) == 118:
        print("Annotated CFD subset: OK (118 image+MAT+SEG pairs)")
    else:
        print("NOTE: annotated pair count differs from the 118-pair subset in the uploaded CFD repository copy.")
    return ok


def check_crack500(path):
    print("\n[Crack500]")
    if not path.is_dir():
        print("ERROR: Crack500 folder does not exist.")
        return False

    images = files_with_ext(path, IMG_EXTS)
    mats = files_with_ext(path, {".mat"})
    print(f"image-like files found recursively: {len(images)}")
    print(f"MAT files found recursively: {len(mats)}")
    top = sorted(p.name for p in path.iterdir() if p.is_dir())
    print("top-level subfolders:", ", ".join(top) if top else "(none)")
    print("NOTE: Crack500 is distributed in multiple directory layouts; exact image-mask pairing will be normalized in the next preprocessing step.")
    ok = len(images) > 0 or len(mats) > 0
    print(f"basic content check: {status(ok)}")
    return ok


print("=" * 72)
print("CrackSSL Dataset Checker")
print("Project:", ROOT)
print("=" * 72)

if not DATA_ROOT.is_dir():
    print(f"ERROR: datasets folder not found: {DATA_ROOT}")
    sys.exit(1)

results = {}
for name in ["Crack500", "DeepCrack", "CFD"]:
    p = DATA_ROOT / name
    if not p.is_dir():
        print(f"\n[{name}]\nERROR: folder not found: {p}")
        results[name] = False
        continue
    if name == "DeepCrack":
        results[name] = check_deepcrack(p)
    elif name == "CFD":
        results[name] = check_cfd(p)
    else:
        results[name] = check_crack500(p)

print("\n" + "=" * 72)
for k, v in results.items():
    print(f"{k:10s}: {status(v)}")
all_ok = all(results.get(x, False) for x in ["Crack500", "DeepCrack", "CFD"])
print("OVERALL   :", status(all_ok))
print("=" * 72)
sys.exit(0 if all_ok else 1)
