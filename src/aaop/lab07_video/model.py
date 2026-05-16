from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CnnLstm(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) #wyciąga cechy
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.lstm = nn.LSTM(512, hidden, 1, batch_first=True) # uczy się zależności czasowych
        self.fc = nn.Linear(hidden, num_classes)
        # blokujemy trening początkowych warstw ResNetu
        for p in self.cnn.parameters():
            p.requires_grad = False
        #ale odmrażamy ostatni blok konwolucyjny
        for p in list(self.cnn.layer4.parameters()):
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W)
        b, c, t, h, w = x.shape
        # trik dla ResNetu, np. 32 filmy po 5 klatek tłumaczymy mu na 160 klatek
        # ponieważ nie rozumie przestrzeni czasowej
        y = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        f = self.cnn(y)
        f = f.view(b, t, -1) # rozklejamy wymiary z powrotem, ponieważ ResNet już je przetworzył
        o, _ = self.lstm(f)
        o = o[:, -1, :] #wybieramy stan wyjściowy tylko z ostatniej klatki
        return self.fc(o)


class FrameBaseline(nn.Module):
    """
    Jedna klatka: ResNet-18, brak analizy czasu
    Weryfikujemy, czy analiza czasu ma znaczenie - porównując ten baseline z CnnLstm
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)
