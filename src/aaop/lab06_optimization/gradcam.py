from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def compute_gradcam(
    model: nn.Module,
    input_tensor: Tensor,
    target_class: int,
    layer: nn.Module,
) -> np.ndarray:
    '''
    Grad-CAM dla warstwy `layer` (np. `resnet.layer4`).
    Wymaga, aby `model` zwracał logity klasyfikacji.
    Sprawdzamy miejsca, które aktywują feature mapy z dużymi pochodnymi dla danej klasy.
    '''
    model.eval()
    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    #forward hook - łapiemy wyjście warstwy w forward passie
    def f_hook(_m, _in, out: Tensor) -> None:
        activations.append(out)

    #backward hook - łapiemy gradienty podczas propagacji wstecznej
    def b_hook(
        _m, _g_in, grad_out: tuple[Tensor, ...] | None
    ) -> None:
        # grad_out to dloss/dlayer_output
        if grad_out is not None and grad_out[0] is not None:
            gradients.append(grad_out[0])

    #hooki nam dają wgląd modelu na dany obrazy
    h1 = layer.register_forward_hook(f_hook) #wstrzykujemy hooki przez pytorch
    h2 = layer.register_full_backward_hook(b_hook)
    with torch.set_grad_enabled(True):
        out = model(input_tensor) # forward pass
        model.zero_grad() #zerujemy gradienty
        score = out[0, target_class] #zapisujemy wynik jako skalar
        score.backward() #dscore/dfeature_maps
    h1.remove() #usuwanie by nie nakładały się w kolejnych wywołaniach
    h2.remove()
    if not activations or not gradients:
        return np.zeros((7, 7), dtype=np.float32)
    A = activations[0][0] # feature mapy z warstwy: bierzemy pierwszy forward pass (1,C,H,W) i usuwamy wymiar batchu (C,H,W)
    dY = gradients[0][0] #gradienty z tej samej warswty
    w = dY.mean(dim=(1, 2)) #liczymy wagi kanałów
    cam = (w.view(-1, 1, 1) * A).sum(0) # ważenie każdego feature mapu A i ich sumowanie
    cam = torch.relu(cam) # pokazujemy tylko te miejsca, gdzie klasa się aktywuje
    #skalujemy
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().cpu().numpy()
    return cam


def overlay_heatmap(
    bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    h, w = bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h)) #resizujemy heatmapę do wymiarów obrazu
    hm_u8 = np.uint8(255 * hm) # 0-1 to 0-255
    #pokolorowanie mapy i dodanie do oryginalnego obrazu
    col = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, col, alpha, 0)
