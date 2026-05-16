"""Augmentacja treningowa przez preset BNNR (fallback: torchvision w dataset)"""

from __future__ import annotations

import warnings

import numpy as np


def bnnr_available() -> bool:
    try:
        import bnnr

        return True
    except ImportError:
        return False


def apply_bnnr_preset_numpy(
    rgb_uint8: np.ndarray,
    preset_name: str,
    random_state: int,
) -> np.ndarray:
    """HWC uint8 RGB → ta sama przestrzeń po łańcuchu augmentacji z presetu BNNR"""
    from bnnr.presets import get_preset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        augs = get_preset(preset_name, random_state=random_state)
    batch = rgb_uint8.astype(np.uint8)[np.newaxis, ...] #dodajemy sztucznie pusty wymiar, bo biblioteka oczekuje takiego wejścia
    out = batch.copy()
    for aug in augs:
        out = aug.apply_batch(out)
    return np.asarray(out[0])


def augment_train_rgb(
    rgb_uint8: np.ndarray,
    *,
    preset_name: str,
    sample_seed: int,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return rgb_uint8
    if not bnnr_available():
        return rgb_uint8
    try:
        return apply_bnnr_preset_numpy(rgb_uint8, preset_name, sample_seed)
    except Exception:
        return rgb_uint8
