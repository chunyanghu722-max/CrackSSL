from pathlib import Path
import argparse, cv2, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from crack500_subset_data import Crack500SubsetDataset, denormalize_image
from baseline_utils import BinarySegmentationMeter, bce_dice_loss, build_model, save_json

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="datasets/Crack500")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--model", choices=["teacher","student"], default="teacher")
    ap.add_argument("--max-images", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    ds = Crack500SubsetDataset(args.data_root, "testcrop", None,
                               int(cfg["height"]), int(cfg["width"]), False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ck["teacher_model" if args.model=="teacher" else "student_model"])
    model.eval()

    meter = BinarySegmentationMeter(threshold=0.5)
    loss_sum = seen = 0.0
    saved = 0
    out = Path(cfg["output"]); pred_dir = out/f"test_predictions_{args.model}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for b in tqdm(dl, desc="Test"):
        x = b["image"].to(device); y = b["mask"].to(device)
        logits = model(x)
        loss,_,_ = bce_dice_loss(logits,y)
        meter.update(logits.float(), y)
        loss_sum += loss.item()*x.size(0); seen += x.size(0)

        if saved < args.max_images:
            preds = (torch.sigmoid(logits).cpu() >= 0.5)
            for i in range(x.size(0)):
                if saved >= args.max_images: break
                img = denormalize_image(b["image"][i])
                gt = (b["mask"][i,0].numpy()>0.5).astype(np.uint8)*255
                pr = preds[i,0].numpy().astype(np.uint8)*255
                gt3 = np.repeat(gt[...,None],3,2)
                pr3 = np.repeat(pr[...,None],3,2)
                ov = img.copy(); m = pr>0
                ov[m] = (0.4*ov[m] + 0.6*np.array([255,0,0])).astype(np.uint8)
                canvas = np.concatenate([img,gt3,pr3,ov],1)
                cv2.imwrite(str(pred_dir/f"{saved+1:02d}_{b['name'][i]}.jpg"),
                            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                saved += 1

    m = meter.compute(); m["loss"] = loss_sum/max(seen,1)
    print("\n"+"="*72)
    print(f"CRACK500 EMA {args.model.upper()} TEST RESULTS")
    print("="*72)
    for k,v in m.items(): print(f"{k:>10s}: {v:.6f}")
    save_json(m, out/f"test_metrics_{args.model}.json")
    print("Saved:", out/f"test_metrics_{args.model}.json")
    print("Saved prediction examples:", pred_dir)

if __name__ == "__main__":
    main()
