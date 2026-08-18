from pathlib import Path
import argparse, copy, math, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from crack500_subset_data import Crack500SubsetDataset
from ema_data import Crack500UnlabeledPairDataset
from baseline_utils import (
    BinarySegmentationMeter, append_csv, bce_dice_loss,
    build_model, save_json, set_seed
)

def rampup(epoch, length):
    if length <= 0: return 1.0
    x = max(0.0, min(float(epoch), float(length)))
    phase = 1.0 - x/length
    return math.exp(-5.0*phase*phase)

def rep_loader(ds, batch, workers, samples, seed):
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(ds, replacement=True, num_samples=samples, generator=g)
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers>0))

def eval_loader(ds, batch, workers):
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers>0))

@torch.no_grad()
def update_teacher(teacher, student, decay, step):
    a = min(1.0 - 1.0/(step+1.0), decay)
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(a).add_(sp.data, alpha=1.0-a)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = BinarySegmentationMeter(threshold=0.5)
    loss_sum, seen = 0.0, 0
    for b in tqdm(loader, desc="Val", leave=False):
        x = b["image"].to(device, non_blocking=True)
        y = b["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type=="cuda")):
            logits = model(x)
            loss,_,_ = bce_dice_loss(logits, y)
        loss_sum += loss.item()*x.size(0); seen += x.size(0)
        meter.update(logits.float(), y)
    m = meter.compute(); m["loss"] = loss_sum/max(seen,1)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/Crack500")
    ap.add_argument("--labeled-list", required=True)
    ap.add_argument("--unlabeled-list", required=True)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-samples", type=int, default=1896)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--max-unsup-weight", type=float, default=1.0)
    ap.add_argument("--rampup-epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output); ckpt = out/"checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out/"config.json")

    lds = Crack500SubsetDataset(args.data_root, "traincrop", args.labeled_list,
                                args.height, args.width, True)
    uds = Crack500UnlabeledPairDataset(args.data_root, args.unlabeled_list,
                                       args.height, args.width)
    vds = Crack500SubsetDataset(args.data_root, "valcrop", None,
                                args.height, args.width, False)
    ll = rep_loader(lds, args.batch_size, args.workers, args.epoch_samples, args.seed)
    ul = rep_loader(uds, args.batch_size, args.workers, args.epoch_samples, args.seed+1)
    vl = eval_loader(vds, args.batch_size, args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*82)
    print("CRACK500 EMA TEACHER-STUDENT BASELINE")
    print("Unique labeled   :", len(lds))
    print("Unique unlabeled :", len(uds))
    print("Updates/epoch    :", len(ll))
    print("EMA decay        :", args.ema_decay)
    print("="*82)

    student = build_model(pretrained=True).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    best, step, history = -1.0, 0, []
    csvp = out/"history.csv"
    if csvp.exists(): csvp.unlink()

    for epoch in range(1, args.epochs+1):
        student.train(); teacher.eval()
        wu = args.max_unsup_weight * rampup(epoch-1, args.rampup_epochs)
        total_sup = total_con = seen = 0.0

        bar = tqdm(zip(ll, ul), total=min(len(ll),len(ul)), desc="Train", leave=False)
        for lb, ub in bar:
            xl = lb["image"].to(device, non_blocking=True)
            yl = lb["mask"].to(device, non_blocking=True)
            uw = ub["weak"].to(device, non_blocking=True)
            us = ub["strong"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type,
                    dtype=torch.float16, enabled=(device.type=="cuda")):
                tprob = torch.sigmoid(teacher(uw)).float()

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                    enabled=(device.type=="cuda")):
                slog_l = student(xl)
                sup,_,_ = bce_dice_loss(slog_l, yl)
                sprob = torch.sigmoid(student(us)).float()
                con = F.mse_loss(sprob, tprob)
                loss = sup + wu*con

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            step += 1
            update_teacher(teacher, student, args.ema_decay, step)

            bs = xl.size(0)
            total_sup += sup.item()*bs
            total_con += con.item()*bs
            seen += bs
            bar.set_postfix(sup=f"{sup.item():.4f}", con=f"{con.item():.5f}")

        sch.step()
        val = validate(teacher, vl, device)
        sup_avg = total_sup/max(seen,1); con_avg = total_con/max(seen,1)
        row = {
            "epoch": epoch, "unsup_weight": wu,
            "train_sup_loss": sup_avg, "train_cons_loss": con_avg,
            "val_loss": val["loss"], "val_dice": val["dice"],
            "val_iou": val["iou"], "val_precision": val["precision"],
            "val_recall": val["recall"], "val_cldice": val["cldice"]
        }
        append_csv(csvp, row); history.append(row)

        state = {
            "epoch": epoch, "student_model": student.state_dict(),
            "teacher_model": teacher.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(), "best_val_dice": best,
            "config": vars(args)
        }
        torch.save(state, ckpt/"last.pt")
        improved = val["dice"] > best
        if improved:
            best = val["dice"]; state["best_val_dice"] = best
            torch.save(state, ckpt/"best.pt")

        print(f"Epoch {epoch:03d}/{args.epochs} | sup {sup_avg:.4f} | con {con_avg:.5f} | "
              f"wu {wu:.3f} | Dice {val['dice']:.4f} | IoU {val['iou']:.4f} | "
              f"P {val['precision']:.4f} | R {val['recall']:.4f} | "
              f"clDice {val['cldice']:.4f}" + ("  <-- BEST" if improved else ""))

    print("\nEMA training finished.")
    print("Best teacher validation Dice:", f"{best:.6f}")
    print("Best checkpoint:", ckpt/"best.pt")

if __name__ == "__main__":
    main()
