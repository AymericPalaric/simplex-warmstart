"""Exploration du simplexe : où sont les meilleures compositions ?

À descripteurs fixés (= les 3 composants d'une étude), on échantillonne le
simplexe des compositions, on prédit, et on décrit la région des meilleures :
son enveloppe convexe, son barycentre, ses bornes par composant.
C'est le point de départ (warm start) d'un plan d'expériences.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .chemistry import N_COMPONENTS
from .features import COMPOSITION_COLS, DESCRIPTOR_COLS, FEATURE_COLS

OBJECTIVES = ("maximize", "minimize")
MIN_ENVELOPE_POINTS = 3  # en dessous, l'enveloppe n'est plus un polygone

# Sommets d'un triangle équilatéral : barycentrique -> plan, pour l'enveloppe convexe
_TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def sample_simplex(n: int, seed: int) -> np.ndarray:
    """n compositions tirées uniformément sur le simplexe -> shape (n, N_COMPONENTS)"""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(N_COMPONENTS), size=n)


def build_frame(compositions: np.ndarray, descriptors: np.ndarray) -> pd.DataFrame:
    """Compositions variables + descripteurs constants -> DataFrame de features.
    compositions.shape = (n, N_COMPONENTS), descriptors.shape = (N_COMPONENTS, N_DESCRIPTORS)"""
    frame = pd.DataFrame(np.asarray(compositions, dtype=float), columns=COMPOSITION_COLS)
    flat = np.asarray(descriptors, dtype=float).reshape(-1)
    for column, value in zip(DESCRIPTOR_COLS, flat, strict=True):
        frame[column] = value
    return frame[FEATURE_COLS]


def convex_hull(points: np.ndarray) -> list[int]:
    """Indices des sommets de l'enveloppe convexe 2D, en sens trigonométrique.
    Monotone chain d'Andrew : O(n log n), pas de dépendance à scipy."""
    order = np.lexsort((points[:, 1], points[:, 0])).tolist()
    if len(order) < MIN_ENVELOPE_POINTS:
        return order

    def turns_right(origin: int, first: int, second: int) -> bool:
        oa = points[first] - points[origin]
        ob = points[second] - points[origin]
        return float(oa[0] * ob[1] - oa[1] * ob[0]) <= 0.0

    hull: list[int] = []
    for chain in (order, order[::-1]):  # bord inférieur puis bord supérieur
        start = len(hull)
        for index in chain:
            while len(hull) - start >= 2 and turns_right(hull[-2], hull[-1], index):
                hull.pop()
            hull.append(index)
        hull.pop()  # le dernier point ouvre la chaîne suivante
    return hull


def region_size(n_samples: int, top_fraction: float) -> int:
    """Nombre de points retenus, borné pour garder une enveloppe exploitable"""
    return int(min(n_samples, max(MIN_ENVELOPE_POINTS, round(n_samples * top_fraction))))


@dataclass(frozen=True)
class RegionSuggestion:
    """Région prometteuse du simplexe, à descripteurs fixés"""

    compositions: np.ndarray  # (n_samples, N_COMPONENTS) points échantillonnés
    predictions: np.ndarray  # (n_samples,) réponses prédites
    top_index: np.ndarray  # indices de la région, du meilleur au moins bon
    envelope_index: np.ndarray  # indices des sommets de l'enveloppe, sens trigo

    @property
    def best_composition(self) -> np.ndarray:
        return self.compositions[self.top_index[0]]

    @property
    def best_prediction(self) -> float:
        return float(self.predictions[self.top_index[0]])

    @property
    def threshold(self) -> float:
        """Réponse du dernier point retenu = frontière de la région"""
        return float(self.predictions[self.top_index[-1]])

    @property
    def region(self) -> np.ndarray:
        return self.compositions[self.top_index]

    @property
    def region_predictions(self) -> np.ndarray:
        return self.predictions[self.top_index]

    @property
    def centroid(self) -> np.ndarray:
        return self.region.mean(axis=0)

    @property
    def envelope(self) -> np.ndarray:
        return self.compositions[self.envelope_index]

    @property
    def bounds(self) -> np.ndarray:
        """(N_COMPONENTS, 2) : min et max de chaque fraction dans la région"""
        region = self.region
        return np.column_stack([region.min(axis=0), region.max(axis=0)])


def suggest_region(
    predict_fn: PredictFn,
    descriptors: np.ndarray,
    n_samples: int = 4000,
    top_fraction: float = 0.05,
    objective: str = "maximize",
    seed: int = 0,
) -> RegionSuggestion:
    """Échantillonne le simplexe, prédit, et renvoie l'enveloppe des meilleurs points"""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    compositions = sample_simplex(n_samples, seed)
    raw = predict_fn(build_frame(compositions, descriptors))
    predictions = np.asarray(raw, dtype=float).ravel()
    if predictions.shape != (n_samples,):
        raise ValueError(f"Expected {n_samples} predictions, got {predictions.shape}")

    scores = predictions if objective == "maximize" else -predictions
    size = region_size(n_samples, top_fraction)
    top = np.argpartition(-scores, size - 1)[:size]
    top = top[np.argsort(-scores[top], kind="stable")]

    envelope = top[convex_hull(compositions[top] @ _TRIANGLE)]

    return RegionSuggestion(
        compositions=compositions,
        predictions=predictions,
        top_index=top,
        envelope_index=envelope,
    )
