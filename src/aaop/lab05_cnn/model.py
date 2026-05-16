from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Prosta architektura CNN do klasyfikacji"""

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # padding = 1 zachowuje rozmiar obrazu po konwolucji
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), #opiekujemy się niestabilnymi gradientami i vanishing/exploding gradients
            nn.ReLU(True),
            nn.MaxPool2d(2), # zmniejszamy rozdzielczość mapy cech dwukrotnie
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), # każdą mapę cech redukujemy do pojedynczej wartości
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(128, num_classes) #fully connected layer, uzyskujemy logity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
