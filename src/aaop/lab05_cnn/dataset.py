from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from aaop.common.bnnr_aug import augment_train_rgb, bnnr_available


def build_transforms(train: bool, size: int = 224) -> transforms.Compose:
    """Kompatybilność wsteczna, czyli pełny łańcuch torchvision (bez BNNR)"""
    return _torchvision_chain(train, size)


def _normalize_only(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ze zbioru ImageNet, odpowiednio średnie i std
            ),
        ]
    )


def _torchvision_chain(train: bool, size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10), #+- 10 stopni
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.0), # jasność, kontrast i nasycenie
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) 
                ),
            ]
        )
    return _normalize_only(size)


def _tensor_from_rgb_pil(
    img: Image.Image,
    *,
    train: bool,
    index: int,
    size: int,
    augment_seed_base: int,
    use_bnnr_augment: bool,
    bnnr_preset: str,
    torchvision_train_tf: transforms.Compose,
    eval_tf: transforms.Compose,
) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    #używamy augmentacji BNNR, jeżeli to możliwe
    if train and use_bnnr_augment and bnnr_available():
        arr = np.asarray(img)
        # ustawiamy losowe augmentacje, ale powtarzalne pomiędzy uruchomieniami
        # 1009 jest pierwszą liczbą, co pomaga ograniczyć powtarzanie wzrowców seedów
        seed_i = augment_seed_base + index * 1009
        arr = augment_train_rgb(
            arr,
            preset_name=bnnr_preset,
            sample_seed=seed_i,
            enabled=True,
        )
        img = Image.fromarray(arr)
        return eval_tf(img) #ToTensor i Normalize
    if train:
        return torchvision_train_tf(img) #+augmentacje tv: flip, rotation, color jitter
    return eval_tf(img)


class ImageFolderList(Dataset):
    """Ładuje ścieżkę, etykietę z listy z CSV/skryptu."""

    def __init__(
        self,
        items: list[tuple[Path, int]],
        train: bool = True, # dla traina stosujemy augmentacje, natomiast dla pozostałych tylko normalizację
        size: int = 224,
        *,
        augment_seed_base: int = 42,
        use_bnnr_augment: bool = True,
        bnnr_preset: str = "light",
    ) -> None:
        self.items = items
        self.train = train
        self.size = size
        self.augment_seed_base = augment_seed_base
        self.use_bnnr_augment = use_bnnr_augment
        self.bnnr_preset = bnnr_preset
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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        '''Zwraca tensor obrazu i etykietę klasy'''
        p, y = self.items[i]
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
        return t, y


def collect_first_frame_per_video(frames_root: Path) -> list[tuple[Path, int]]:
    '''Wybieramy po jednym framie z każdego video'''
    class_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    out: list[tuple[Path, int]] = []
    for ci, cdir in enumerate(class_dirs):
        for vdir in sorted(cdir.iterdir()):
            if not vdir.is_dir():
                continue
            jpgs = sorted(vdir.glob("*.jpg"))
            jpg = jpgs[0] if jpgs else None
            if jpg is not None:
                out.append((jpg, ci))
    return out
