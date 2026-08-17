"""Baseline : MLP sur (compositions, descripteurs)"""

from __future__ import annotations

import torch
from torch import nn


class MixtureMLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int] = (64, 64), dropout: float = 0.0):
        super().__init__()
        layers = []
        previous = n_features
        for width in hidden:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
