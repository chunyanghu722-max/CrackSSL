from pathlib import Path
import argparse
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


def replacement_loader(dataset, batch_size, workers, epoch_samples, seed):
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=epoch_samples, generator=g)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers > 0), drop_last=False)


def eval_loader(dataset, batch_size, workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers > 0), drop_last=False)


def hard_binary_pseudo(logits):
    return (torch.sigmoid(logits.detach()) >= 0.5).float()


def cross_pseudo_bce(logits_a, logits_b):
    """Bidirectional CPS term for a binary task."""
    pseudo_a = hard_binary_pseudo(logits_a)
    pseudo_b = hard_binary_pseudo(logits_b)
    loss_b_from_a = F.binary_cross_entropy_with_logits(logits_b, pseudo_a)
    loss_a_from_b = F.binary_cross_entropy_with_logits(logits_a, pseudo_b)
    return 0.5 * (loss_b_from_a + loss_a_from_b)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = BinarySegmentationMeter(threshold=0.5, compute_cldice=True)
    loss_sum = 0.0
    seen = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
            logits = model(x)
            loss, _, _ = bce_dice_loss(logits, y)
        loss_sum += loss.item() * x.size(0)
        seen += x.size(0)
        meter.update(logits.float(), y)
    m = meter.compute()
    m["loss"] = loss_sum / max(seen, 1)
    return m


def main():
    ap = argparse.ArgumentParser(description="Crack500 CPS baseline (binary adaptation)")
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
    ap.add_argument("--cps-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), out / "config.json")

    lds = Crack500SubsetDataset(args.data_root, "traincrop", args.labeled_list,
                                args.height, args.width, True)
    uds = Crack500UnlabeledPairDataset(args.data_root, args.unlabeled_list,
                                       args.height, args.width)
    vds = Crack500SubsetDataset(args.data_root, "valcrop", None,
                                args.height, args.width, False)
    ll = replacement_loader(lds, args.batch_size, args.workers,
                            args.epoch_samples, args.seed)
    ul = replacement_loader(uds, args.batch_size, args.workers,
                            args.epoch_samples, args.seed + 1)
    vl = eval_loader(vds, args.batch_size, args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("CRACK500 CPS BASELINE | same DeepLabV3+/ResNet50 | binary cross pseudo supervision")
    print("Labeled:", len(lds), "Unlabeled:", len(uds), "Updates/epoch:", len(ll))
    print("CPS weight:", args.cps_weight, "Seed:", args.seed)
    print("Evaluation branch: Network-1 only (no ensemble)")
    print("=" * 88)

    # Two identically structured but independently initialized networks.
    # ImageNet encoder initialization is the same source; randomly initialized heads differ.
    net1 = build_model(pretrained=True).to(device)
    net2 = build_model(pretrained=True).to(device)

    opt = torch.optim.AdamW(list(net1.parameters()) + list(net2.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best = -1.0
    history_path = out / "history.csv"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(1, args.epochs + 1):
        net1.train(); net2.train()
        sup_sum = cps_l_sum = cps_u_sum = 0.0
        seen = 0
        bar = tqdm(zip(ll, ul), total=min(len(ll), len(ul)), desc="CPS Train", leave=False)
        for lb, ub in bar:
            xl = lb["image"].to(device, non_blocking=True)
            yl = lb["mask"].to(device, non_blocking=True)
            # CPS uses the same augmented input for both networks.
            # We use the existing strong photometric view for the common unlabeled input.
            xu = ub["strong"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # Labeled group: supervised losses + CPS on labeled data, as in the CPS paper.
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                l1 = net1(xl)
                l2 = net2(xl)
                sup1, _, _ = bce_dice_loss(l1, yl)
                sup2, _, _ = bce_dice_loss(l2, yl)
                sup = 0.5 * (sup1 + sup2)
                cps_l = cross_pseudo_bce(l1, l2)
                labeled_loss = sup + args.cps_weight * cps_l
            scaler.scale(labeled_loss).backward()

            # Unlabeled group: bidirectional cross pseudo supervision.
            # Backward is separated from the labeled group to reduce peak activation memory;
            # optimizer.step() is still performed once after both gradients are accumulated.
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                u1 = net1(xu)
                u2 = net2(xu)
                cps_u = cross_pseudo_bce(u1, u2)
                unlabeled_loss = args.cps_weight * cps_u
            scaler.scale(unlabeled_loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = xl.size(0)
            seen += bs
            sup_sum += sup.item() * bs
            cps_l_sum += cps_l.item() * bs
            cps_u_sum += cps_u.item() * bs
            bar.set_postfix(sup=f"{sup.item():.4f}", cps_u=f"{cps_u.item():.4f}")

        sch.step()
        val1 = validate(net1, vl, device)
        val2 = validate(net2, vl, device)
        improved = val1["dice"] > best
        if improved:
            best = val1["dice"]

        row = {
            "epoch": epoch,
            "train_sup_loss": sup_sum / max(seen, 1),
            "train_cps_l_loss": cps_l_sum / max(seen, 1),
            "train_cps_u_loss": cps_u_sum / max(seen, 1),
            "val1_dice": val1["dice"], "val1_iou": val1["iou"],
            "val1_precision": val1["precision"], "val1_recall": val1["recall"],
            "val1_cldice": val1["cldice"], "val1_loss": val1["loss"],
            "val2_dice": val2["dice"], "val2_iou": val2["iou"],
            "val2_precision": val2["precision"], "val2_recall": val2["recall"],
            "val2_cldice": val2["cldice"], "val2_loss": val2["loss"],
        }
        append_csv(history_path, row)

        state = {
            "epoch": epoch,
            "model1": net1.state_dict(),
            "model2": net2.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "best_val_dice": best,
            "config": vars(args),
        }
        torch.save(state, ckpt_dir / "last.pt")
        if improved:
            torch.save(state, ckpt_dir / "best.pt")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | sup {row['train_sup_loss']:.4f} | "
            f"cpsL {row['train_cps_l_loss']:.4f} | cpsU {row['train_cps_u_loss']:.4f} | "
            f"Net1 Dice {val1['dice']:.4f} IoU {val1['iou']:.4f} P {val1['precision']:.4f} "
            f"R {val1['recall']:.4f} clDice {val1['cldice']:.4f}" + ("  <-- BEST" if improved else "")
        )

    print("\nCPS finished. Best Network-1 validation Dice:", f"{best:.6f}")
    print("Checkpoint:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
