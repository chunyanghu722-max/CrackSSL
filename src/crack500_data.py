from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_pair(image, mask, target_h: int, target_w: int):
    """
    Preserve aspect ratio, resize to fit inside target_h x target_w,
    then symmetrically zero-pad to exactly target_h x target_w.
    """
    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    image = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    mask = cv2.resize(
        mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    pad_h = target_h - new_h
    pad_w = target_w - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    mask = cv2.copyMakeBorder(
        mask, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    return image, mask


def build_transform(train: bool):
    ops = []
    if train:
        ops.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.30,
            ),
        ])

    ops.append(
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        )
    )
    return A.Compose(ops)


class Crack500Dataset(Dataset):
    """
    CRACK500 crop structure:
        split/*.jpg = image
        split/*.png = binary mask with same stem
    """
    def __init__(
        self,
        root_dir,
        split="traincrop",
        height=384,
        width=640,
        train=False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.height = int(height)
        self.width = int(width)
        self.train = bool(train)

        if self.height % 16 != 0 or self.width % 16 != 0:
            raise ValueError(
                f"Target size must be divisible by 16, got "
                f"{self.height}x{self.width}"
            )

        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {self.split_dir}")

        images = sorted(self.split_dir.glob("*.jpg"))
        if not images:
            raise RuntimeError(f"No JPG files found in: {self.split_dir}")

        self.pairs = []
        missing = []
        for image_path in images:
            mask_path = image_path.with_suffix(".png")
            if mask_path.exists():
                self.pairs.append((image_path, mask_path))
            else:
                missing.append(image_path.name)

        if missing:
            raise RuntimeError(
                f"{len(missing)} images do not have matching PNG masks. "
                f"Examples: {missing[:5]}"
            )

        self.transform = build_transform(train=self.train)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preserve geometry; do not stretch CRACK500 samples.
        image, mask = letterbox_pair(
            image,
            mask,
            target_h=self.height,
            target_w=self.width,
        )

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "name": image_path.stem,
        }


def denormalize_image(tensor):
    x = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    x = (x * std + mean) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)
