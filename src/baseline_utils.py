from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

try:
    from skimage.morphology import skeletonize
except ImportError as e:
    raise ImportError(
        "clDice evaluation requires scikit-image. "
        "Install it with: pip install scikit-image"
    ) from e


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained=True):
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def bce_dice_loss(logits, targets, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    dice_loss = 1.0 - dice.mean()

    total = bce + dice_loss
    return total, bce.detach(), dice_loss.detach()


def _cldice_one(pred_bool: np.ndarray, target_bool: np.ndarray) -> float:
    """
    Hard clDice for one binary prediction/ground-truth pair.

    clDice = harmonic mean of:
      topology precision   = |S(pred) ∩ target| / |S(pred)|
      topology sensitivity = |S(target) ∩ pred| / |S(target)|

    Empty-mask handling:
      - both empty -> 1.0
      - exactly one empty -> 0.0
    """
    pred_bool = np.asarray(pred_bool, dtype=bool)
    target_bool = np.asarray(target_bool, dtype=bool)

    pred_sum = int(pred_bool.sum())
    target_sum = int(target_bool.sum())

    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0

    pred_skel = skeletonize(pred_bool)
    target_skel = skeletonize(target_bool)

    pred_skel_n = int(pred_skel.sum())
    target_skel_n = int(target_skel.sum())

    if pred_skel_n == 0 or target_skel_n == 0:
        return 0.0

    tprec = float(np.logical_and(pred_skel, target_bool).sum()) / pred_skel_n
    tsens = float(np.logical_and(target_skel, pred_bool).sum()) / target_skel_n

    denom = tprec + tsens
    if denom <= 0.0:
        return 0.0

    return float(2.0 * tprec * tsens / denom)


class BinarySegmentationMeter:
    """
    Pixel metrics are accumulated globally across the dataset.
    clDice is computed per image and averaged across images.
    """
    def __init__(self, threshold=0.5, compute_cldice=True):
        self.threshold = threshold
        self.compute_cldice = compute_cldice
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.cldice_sum = 0.0
        self.cldice_count = 0

    @torch.no_grad()
    def update(self, logits, targets):
        probs = torch.sigmoid(logits)
        preds = probs >= self.threshold
        target = targets >= 0.5

        self.tp += torch.logical_and(preds, target).sum().item()
        self.fp += torch.logical_and(preds, ~target).sum().item()
        self.fn += torch.logical_and(~preds, target).sum().item()
        self.tn += torch.logical_and(~preds, ~target).sum().item()

        if self.compute_cldice:
            pred_np = preds.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            for i in range(pred_np.shape[0]):
                pred_mask = pred_np[i, 0]
                target_mask = target_np[i, 0]
                self.cldice_sum += _cldice_one(pred_mask, target_mask)
                self.cldice_count += 1

    def compute(self):
        eps = 1e-8
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        cldice = (
            self.cldice_sum / self.cldice_count
            if self.cldice_count > 0
            else float("nan")
        )

        return {
            "dice": dice,
            "f1": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "cldice": cldice,
            "accuracy": accuracy,
        }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
