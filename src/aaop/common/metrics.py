from __future__ import annotations


def topk_accuracy(logits, targets, k: int = 5) -> float:  # torch.Tensor
    """sprawdzamy, czy prawidłowa odpowiedź znalazła się w gronie K najlepszych typowań modelu"""
    import torch

    k = min(k, logits.size(1))
    _, pred = logits.topk(k, dim=1)
    t = targets.view(-1, 1) # (32,) -> (32,1)
    correct = (pred == t).any(dim=1) #porównujemy całe batche
    return float(correct.float().mean().item())


def iou_xywh(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """IoU dla boxów (x, y, w, h) w pikselach, lewy górny róg"""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_x2, a_y2 = ax + aw, ay + ah # współrzędne prawego dolnego rogu
    b_x2, b_y2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    inter_w = max(0.0, inter_x2 - inter_x1) #zabezpieczamy przed przypadkiem, gdy ramki w ogóle się nie nakładają
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)


def confusion_by_class(y_true, y_pred, num_classes: int):
    import numpy as np

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1 #zliczamy wystąpienia dla poszczególnych przypadków
    return cm
