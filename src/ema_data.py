from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

def letterbox(image, target_h=384, target_w=640):
    h, w = image.shape[:2]
    s = min(target_h / h, target_w / w)
    nh, nw = max(1, round(h*s)), max(1, round(w*s))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = target_h - nh, target_w - nw
    t, b = ph//2, ph-ph//2
    l, r = pw//2, pw-pw//2
    return cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value=(0,0,0))

def norm_tensor(image):
    x = image.astype(np.float32)/255.0
    x = (x - MEAN)/STD
    return torch.from_numpy(x.transpose(2,0,1)).float()

def strong_photo(image):
    out = image.copy()
    if random.random() < 0.8:
        out = cv2.convertScaleAbs(
            out,
            alpha=random.uniform(0.75, 1.25),
            beta=random.uniform(-20, 20),
        )
    if random.random() < 0.25:
        out = cv2.GaussianBlur(out, (3,3), 0)
    if random.random() < 0.25:
        noise = np.random.normal(0, 6, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    return out

class Crack500UnlabeledPairDataset(Dataset):
    def __init__(self, root_dir, list_file, height=384, width=640):
        self.root = Path(root_dir) / "traincrop"
        self.height, self.width = int(height), int(width)
        stems = [x.strip() for x in Path(list_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.images = [self.root / f"{s}.jpg" for s in stems]
        missing = [str(p) for p in self.images if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing unlabeled images: {missing[:5]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        p = self.images[idx]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox(img, self.height, self.width)

        # same geometry for teacher and student
        if random.random() < 0.5:
            img = np.ascontiguousarray(np.flip(img, 1))
        if random.random() < 0.5:
            img = np.ascontiguousarray(np.flip(img, 0))

        weak = img.copy()
        strong = strong_photo(img)
        return {"weak": norm_tensor(weak), "strong": norm_tensor(strong), "name": p.stem}
