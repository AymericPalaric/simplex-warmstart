"""Construit les features pour ML"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

COMPOSITION_COLS = ["x1", "x2", "x3"]
COMPOSITION_IDX = tuple(range(3))
DESCRIPTOR_COLS = [f"c{i + 1}_d{j + 1}" for i in range(3) for j in range(2)]
DESCRIPTOR_IDX = tuple(tuple(3 + 2 * c + d for c in range(3)) for d in range(2))
FEATURE_COLS = COMPOSITION_COLS + DESCRIPTOR_COLS
SYMMETRY_GROUPS = (COMPOSITION_IDX, *DESCRIPTOR_IDX)  # Cols qui doivent avoir la même norm
COMPONENT_PERMUTATIONS = tuple(permutations(range(3)))

TARGET_COL = "y"

# Colonnes indispo normalement en prod
LEAKY_COLS = ("y_true",)


def _permutation_index(perm: tuple[int, ...]) -> list[int]:
    index = [perm[slot] for slot in range(3)]
    for slot in range(3):
        source = perm[slot]
        index.extend([3 + 2 * source, 3 + 2 * source + 1])
    return index


PERMUTATION_INDEX = np.asarray(
    [_permutation_index(p) for p in COMPONENT_PERMUTATIONS], dtype=np.int64
)


def augment_permutations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Répète chaque sample avec toutes les permut équivalentes"""
    x_aug = np.concatenate([x[:, index] for index in PERMUTATION_INDEX], axis=0)
    y_aug = np.tile(y, (len(PERMUTATION_INDEX), 1))
    return x_aug, y_aug


def build_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Construit X et y à partir d'un DataFrame d'expériences"""
    leaks = sorted(set(FEATURE_COLS) & set(LEAKY_COLS))
    if leaks:
        raise ValueError(f"Leaky columns in features: {leaks}")
    X = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = frame[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y


@dataclass(frozen=True)
class Standardizer:
    """Standardise les features; train uniquement"""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        groups: tuple[tuple, ...] | None = None,
    ) -> Standardizer:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        if groups is not None:
            for group in groups:
                columns = list(group)
                mean[columns] = x[:, columns].mean()
                std[columns] = x[:, columns].std()
        std[std < 1e-8] = 1.0  # ZeroDivisionError
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Standardizer:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )
