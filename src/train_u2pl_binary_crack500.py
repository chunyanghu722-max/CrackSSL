from pathlib import Path
import argparse, copy
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from crack500_subset_data import Crack500SubsetDataset
from ema_data import Crack500UnlabeledPairDataset
from baseline_utils import BinarySegmentationMeter, append_csv, bce_dice_loss, build_model, save_json, set_seed
from u2pl_binary_utils import (
    ProjectionHead, CPUFeatureQueue, forward_with_decoder_features, infer_decoder_channels,
    binary_distribution_from_logit, adaptive_entropy_masks, masked_pseudo_bce,
    u2pl_binary_contrastive,
)


def replacement_loader(ds, batch, workers, samples, seed):
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(ds, replacement=True, num_samples=samples, generator=g)
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers > 0), drop_last=False)


def eval_loader(ds, batch, workers):
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers > 0), drop_last=False)


@torch.no_grad()
def update_ema_module(teacher, student, decay, step):
    a = min(1.0 - 1.0 / (step + 1.0), float(decay))
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(a).add_(sp.data, alpha=1.0 - a)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = BinarySegmentationMeter(threshold=0.5, compute_cldice=True)
    loss_sum = 0.0; seen = 0
    for b in tqdm(loader, desc="Val", leave=False):
        x = b["image"].to(device, non_blocking=True)
        y = b["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
            logits = model(x)
            loss, _, _ = bce_dice_loss(logits, y)
        meter.update(logits.float(), y)
        loss_sum += loss.item() * x.size(0); seen += x.size(0)
    m = meter.compute(); m["loss"] = loss_sum / max(seen, 1)
    return m


def main():
    ap = argparse.ArgumentParser(description="Crack500 U2PL binary-task adaptation")
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
    ap.add_argument("--sup-only-epochs", type=int, default=1)
    ap.add_argument("--initial-unreliable", type=float, default=0.20)
    ap.add_argument("--unsup-weight", type=float, default=1.0)
    ap.add_argument("--contrast-weight", type=float, default=0.1)
    ap.add_argument("--projection-dim", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--num-queries", type=int, default=32)
    ap.add_argument("--num-negatives", type=int, default=64)
    ap.add_argument("--queue-size-bg", type=int, default=12000)
    ap.add_argument("--queue-size-crack", type=int, default=6000)
    ap.add_argument("--anchor-threshold", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output); ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["u2pl_binary_adaptation"] = "Original U2PL PRT [3,20) is invalid for C=2; adapted to the only non-top rank [1,2)."
    save_json(cfg, out / "config.json")

    lds = Crack500SubsetDataset(args.data_root, "traincrop", args.labeled_list,
                                args.height, args.width, True)
    uds = Crack500UnlabeledPairDataset(args.data_root, args.unlabeled_list,
                                       args.height, args.width)
    vds = Crack500SubsetDataset(args.data_root, "valcrop", None,
                                args.height, args.width, False)
    ll = replacement_loader(lds, args.batch_size, args.workers, args.epoch_samples, args.seed)
    ul = replacement_loader(uds, args.batch_size, args.workers, args.epoch_samples, args.seed + 1)
    vl = eval_loader(vds, args.batch_size, args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_model(pretrained=True).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()

    dec_ch = infer_decoder_channels(student, args.height, args.width, device)
    projector_s = ProjectionHead(dec_ch, args.projection_dim).to(device)
    projector_t = copy.deepcopy(projector_s).to(device)
    for p in projector_t.parameters(): p.requires_grad_(False)
    projector_t.eval()

    queues = [CPUFeatureQueue(args.queue_size_bg), CPUFeatureQueue(args.queue_size_crack)]

    opt = torch.optim.AdamW(list(student.parameters()) + list(projector_s.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("=" * 92)
    print("CRACK500 U2PL BINARY-TASK ADAPTATION")
    print("Original U2PL uses multi-class PRT [3,20). Crack/background has C=2, so")
    print("the PRT is adapted to [1,2): an unreliable pixel is negative for its rank-2 class.")
    print("This must be disclosed as a binary adaptation if reported in the paper.")
    print("Labeled:", len(lds), "Unlabeled:", len(uds), "Seed:", args.seed)
    print("=" * 92)

    best = -1.0; step = 0
    history = out / "history.csv"
    if history.exists(): history.unlink()

    for epoch in range(1, args.epochs + 1):
        student.train(); projector_s.train(); teacher.eval(); projector_t.eval()
        if epoch == args.sup_only_epochs + 1:
            # Official U2PL initializes the teacher from the trained student after supervised warm-up.
            teacher.load_state_dict(student.state_dict())
            projector_t.load_state_dict(projector_s.state_dict())
            step = 0
        sup_sum = uns_sum = con_sum = 0.0; seen = 0; alpha_sum = 0.0
        bar = tqdm(zip(ll, ul), total=min(len(ll), len(ul)), desc="U2PL Train", leave=False)
        for lb, ub in bar:
            xl = lb["image"].to(device, non_blocking=True)
            yl = lb["mask"].to(device, non_blocking=True)
            uw = ub["weak"].to(device, non_blocking=True)
            us = ub["strong"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                slog_l, sfeat_l = forward_with_decoder_features(student, xl)
                sup, _, _ = bce_dice_loss(slog_l, yl)
                sz_l = projector_s(sfeat_l)

            if epoch <= args.sup_only_epochs:
                uns = 0.0 * sup
                con = 0.0 * sup
                scaler.scale(sup).backward()
                scaler.step(opt); scaler.update()
                step += 1
                # The teacher is not used during supervised warm-up; it is copied from the
                # student once semi-supervised training starts (matching original U2PL logic).
                alpha_t = args.initial_unreliable
            else:
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(device.type == "cuda")):
                    tlog_u, tfeat_u = forward_with_decoder_features(teacher, uw)
                    tlog_l, tfeat_l = forward_with_decoder_features(teacher, xl)
                    tz_u = projector_t(tfeat_u)
                    tz_l = projector_t(tfeat_l)
                    reliable, unreliable, _, alpha_t = adaptive_entropy_masks(
                        tlog_u.float(), epoch - 1, args.epochs, args.initial_unreliable)

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=(device.type == "cuda")):
                    slog_u, sfeat_u = forward_with_decoder_features(student, us)
                    sz_u = projector_s(sfeat_u)
                    uns = masked_pseudo_bce(slog_u.float(), tlog_u.float(), reliable)
                    con = u2pl_binary_contrastive(
                        sz_l, sz_u, tz_l, tz_u, yl, tlog_l.float(), tlog_u.float(), reliable, unreliable,
                        queues, current_class_threshold=args.anchor_threshold,
                        num_queries=args.num_queries, num_negatives=args.num_negatives,
                        temperature=args.temperature,
                    )
                    loss = sup + args.unsup_weight * uns + args.contrast_weight * con
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                step += 1
                update_ema_module(teacher, student, args.ema_decay, step)
                update_ema_module(projector_t, projector_s, args.ema_decay, step)

            bs = xl.size(0); seen += bs
            sup_sum += sup.item() * bs
            uns_sum += float(uns.detach().item()) * bs
            con_sum += float(con.detach().item()) * bs
            alpha_sum += float(alpha_t) * bs
            bar.set_postfix(sup=f"{sup.item():.4f}", uns=f"{float(uns.detach()):.4f}", con=f"{float(con.detach()):.4f}")

        sch.step()
        val = validate(teacher, vl, device)
        improved = val["dice"] > best
        if improved: best = val["dice"]
        row = {
            "epoch": epoch,
            "train_sup_loss": sup_sum / max(seen, 1),
            "train_unsup_loss": uns_sum / max(seen, 1),
            "train_contrast_loss": con_sum / max(seen, 1),
            "unreliable_fraction": alpha_sum / max(seen, 1),
            "queue_bg": len(queues[0]), "queue_crack": len(queues[1]),
            "val_loss": val["loss"], "val_dice": val["dice"], "val_iou": val["iou"],
            "val_precision": val["precision"], "val_recall": val["recall"], "val_cldice": val["cldice"],
        }
        append_csv(history, row)
        state = {
            "epoch": epoch, "student_model": student.state_dict(), "teacher_model": teacher.state_dict(),
            "student_projector": projector_s.state_dict(), "teacher_projector": projector_t.state_dict(),
            "queue_bg": queues[0].state_dict(), "queue_crack": queues[1].state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sch.state_dict(), "best_val_dice": best,
            "config": cfg,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if improved: torch.save(state, ckpt_dir / "best.pt")
        print(f"Epoch {epoch:03d}/{args.epochs} | sup {row['train_sup_loss']:.4f} | uns {row['train_unsup_loss']:.4f} | "
              f"con {row['train_contrast_loss']:.4f} | unreliable {row['unreliable_fraction']:.3f} | "
              f"Dice {val['dice']:.4f} IoU {val['iou']:.4f} P {val['precision']:.4f} R {val['recall']:.4f} "
              f"clDice {val['cldice']:.4f}" + ("  <-- BEST" if improved else ""))

    print("\nU2PL binary adaptation finished. Best teacher validation Dice:", f"{best:.6f}")
    print("Checkpoint:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
