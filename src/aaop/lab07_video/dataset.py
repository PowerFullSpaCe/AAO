from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms


def read_clip_frames(
    path: str | Path, num_frames: int, stride: int, size: int = 224
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return torch.zeros(3, num_frames, size, size)
    idxs: list[int] = []
    if n <= 1:
        idxs = [0] * num_frames
    else:
        for i in range(num_frames):
            j = int(i * (n - 1) / max(num_frames - 1, 1)) # uzyskanie równomiernej liczby klatek
            idxs.append(j)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
    tfs = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    frs = []
    for j in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(j))
        ok, fr = cap.read()
        if not ok or fr is None:
            fr = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frs.append(tfs(fr))
    cap.release()
    # (T, C, H, W) -> (C, T, H, W)
    return torch.stack(frs, dim=1).float()


class VideoClipDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[Path, int]],
        num_frames: int,
        stride: int = 1,
    ) -> None:
        self.rows = rows
        self.nf = num_frames
        self.stride = stride

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        p, y = self.rows[i]
        x = read_clip_frames(p, self.nf, self.stride)
        return x, y


def single_frame_from_clip(clip: torch.Tensor) -> torch.Tensor:
    """
    Środkowa klatka: (3,H,W) baseline.
    Przyjmujemy tensor (C, T, H, W) i wycinamy z niego środkową klatkę T//2
    """
    return clip[:, clip.size(1) // 2, :, :].clone() # tworzymy niezależny obiekt w pamięci
