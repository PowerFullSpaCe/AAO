from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aaop.lab05_cnn.dataset import _tensor_from_rgb_pil
from torchvision import transforms


class BBoxImageList(Dataset):
    '''Podobnie jak w L5, tylko dodatkowo zwracamy bbox'''
    def __init__(
        self,
        items: list[tuple[Path, int]],
        train: bool,
        *,
        augment_seed_base: int = 42,
        use_bnnr_augment: bool = True,
        bnnr_preset: str = "light",
        size: int = 224,
    ) -> None:
        self.items = items
        self.train = train
        self.augment_seed_base = augment_seed_base
        self.use_bnnr_augment = use_bnnr_augment
        self.bnnr_preset = bnnr_preset
        self.size = size
        self.eval_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        self.tv_train_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        p, y = self.items[i]
        box = largest_bbox_normalized(p)
        img = Image.open(p).convert("RGB")
        t = _tensor_from_rgb_pil(
            img,
            train=self.train,
            index=i,
            size=self.size,
            augment_seed_base=self.augment_seed_base,
            use_bnnr_augment=self.use_bnnr_augment,
            bnnr_preset=self.bnnr_preset,
            torchvision_train_tf=self.tv_train_tf,
            eval_tf=self.eval_tf,
        )
        return t, y, torch.from_numpy(box)


def largest_bbox_normalized(path: str | Path) -> np.ndarray:
    '''Wykrywamy największy obiekt na obrazie i rysujemy jego bounding box'''
    p = str(path)
    bgr = cv2.imread(p)
    if bgr is None:
        return np.array([0.1, 0.1, 0.5, 0.5], dtype=np.float32)
    h0, w0 = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(th.mean()) > 127:
        th = 255 - th
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    c = max(cnts, key=cv2.contourArea) # zakładamy, że największy kontur to człowiek bądź obiekt
    x, y, w, h = cv2.boundingRect(c)
    return np.array(
        [x / w0, y / h0, w / w0, h / h0], dtype=np.float32 # bbox jest niezależny od rozdzielczości obrazu, my go normalizujemy
    )
