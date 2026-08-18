from pathlib import Path
import csv, json, math

ROOT = Path("results")

configs = [
    (0.02, 0.005, ROOT / "sens_skel002_con0005_seed42"),
    (0.02, 0.010, ROOT / "sens_skel002_con001_seed42"),
    (0.02, 0.020, ROOT / "sens_skel002_con002_seed42"),
    (0.05, 0.005, ROOT / "sens_skel005_con0005_seed42"),
    (0.05, 0.010, ROOT / "contrast_seed42_10pct"),  # official existing Full run
    (0.05, 0.020, ROOT / "sens_skel005_con002_seed42"),
    (0.10, 0.005, ROOT / "sens_skel010_con0005_seed42"),
    (0.10, 0.010, ROOT / "sens_skel010_con001_seed42"),
    (0.10, 0.020, ROOT / "sens_skel010_con002_seed42"),
]

metrics_order = ["dice", "iou", "precision", "recall", "cldice", "accuracy", "loss"]

rows = []
for skel, con, folder in configs:
    f = folder / "test_metrics_teacher.json"
    if not f.exists():
        rows.append({
            "lambda_skel": skel,
            "lambda_con": con,
            "folder": str(folder),
            "status": "MISSING",
        })
        continue
    with f.open("r", encoding="utf-8") as fh:
        m = json.load(fh)
    row = {
        "lambda_skel": skel,
        "lambda_con": con,
        "folder": str(folder),
        "status": "OK",
    }
    for k in metrics_order:
        row[k] = float(m[k])
    rows.append(row)

# Full table
fields = ["lambda_skel", "lambda_con"] + metrics_order + ["folder", "status"]
with open("sensitivity_results_seed42.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

# Dice matrix
skels = [0.02, 0.05, 0.10]
cons = [0.005, 0.010, 0.020]
lookup = {(r["lambda_skel"], r["lambda_con"]): r for r in rows}

with open("sensitivity_dice_matrix_seed42.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["lambda_skel \\ lambda_con"] + cons)
    for s in skels:
        vals = []
        for c in cons:
            r = lookup[(s, c)]
            vals.append("" if r["status"] != "OK" else f'{r["dice"]:.6f}')
        w.writerow([s] + vals)

valid = [r for r in rows if r["status"] == "OK"]
valid_sorted = sorted(valid, key=lambda r: r["dice"], reverse=True)

lines = []
lines.append("Crack500 sensitivity analysis: seed42, 10% labeled")
lines.append("=" * 72)
lines.append("")
for r in rows:
    if r["status"] == "OK":
        lines.append(
            f'lambda_skel={r["lambda_skel"]:.3f}, lambda_con={r["lambda_con"]:.3f} | '
            f'Dice={r["dice"]:.6f}, IoU={r["iou"]:.6f}, '
            f'P={r["precision"]:.6f}, R={r["recall"]:.6f}, '
            f'clDice={r["cldice"]:.6f}, Acc={r["accuracy"]:.6f}, Loss={r["loss"]:.6f}'
        )
    else:
        lines.append(
            f'lambda_skel={r["lambda_skel"]:.3f}, lambda_con={r["lambda_con"]:.3f} | MISSING'
        )

lines.append("")
if valid_sorted:
    best = valid_sorted[0]
    lines.append(
        f'Best Dice: {best["dice"]:.6f} at '
        f'lambda_skel={best["lambda_skel"]}, lambda_con={best["lambda_con"]}'
    )
    central = lookup.get((0.05, 0.01))
    if central and central["status"] == "OK":
        lines.append(
            f'Central setting (0.05, 0.01): Dice={central["dice"]:.6f}, '
            f'IoU={central["iou"]:.6f}, clDice={central["cldice"]:.6f}'
        )

Path("sensitivity_summary_seed42.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
