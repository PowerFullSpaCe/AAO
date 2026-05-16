from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models



def resnet18_classifier(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes) #zamieniamy klasyfikator z 512->1000 (ImageNet) na 512-> <nasza_liczba_klas>
    return m


class ResNetBBox(nn.Module):
    """ResNet-18: klasyfikacja + regresja bbox (x, y, w, h) w [0, 1]"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        b = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = b.fc.in_features
        b.fc = nn.Identity() # robimy 512->512 zamiast 512->1000
        self.backbone = b
        self.cls = nn.Linear(in_f, num_classes) # znowu daje logity klas
        self.box = nn.Linear(in_f, 4) #dane bboxa

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x) #wynik to (B,512)
        return self.cls(f), torch.sigmoid(self.box(f))
