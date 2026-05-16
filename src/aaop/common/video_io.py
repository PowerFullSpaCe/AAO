from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frame_bgr(path: str | Path, index: int = 0) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, index) # przewijamy video do klatki o odpowiednim indeksie
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def sample_frame_indices(num_frames: int, n: int) -> list[int]:
    '''Implementacja równomiernego próbkowania klatek'''
    if num_frames <= 0:
        return []
    if n >= num_frames:
        return list(range(num_frames))
    return [int(i * (num_frames - 1) / (n - 1)) for i in range(n)]


def count_frames(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #zczytujemy liczbę klatek danego video
    cap.release()
    return n
