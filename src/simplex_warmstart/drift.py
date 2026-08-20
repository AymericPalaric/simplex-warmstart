"""Détection d'OOD"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

PSI_BINS = 10
EPSILON = 1e-6


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = PSI_BINS) -> float:
    """PSI = divergence entre 2 histo"""
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), EPSILON, None)
    cur_share = np.clip(np.histogram(cur, bins=edges)[0] / len(cur), EPSILON, None)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_stat(ref: np.ndarray, cur: np.ndarray) -> float:
    """Stat de Kolmogorov-Smirnov = gap max entre les 2 CDF"""
    ref_sorted, cur_sorted = np.sort(ref), np.sort(cur)
    grid = np.concatenate([ref_sorted, cur_sorted])
    ref_cdf = np.searchsorted(ref_sorted, grid, side="right") / len(ref_sorted)
    cur_cdf = np.searchsorted(cur_sorted, grid, side="right") / len(cur_sorted)

    return float(np.max(np.abs(ref_cdf - cur_cdf)))


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    psi: float
    ks: float
    drifted: bool


def new_categories(ref: pd.Series, cur: pd.Series) -> list[str]:
    return sorted(set(cur.unique()) - set(ref.unique()))


def compute_drift(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    columns: list[str],
    psi_thresh: float = 0.2,
    share_thresh: float = 0.4,
    categorical_columns: tuple[str, ...] = ("family", "protocol"),
) -> dict:
    results = [
        ColumnDrift(
            column=column,
            psi=(psi := population_stability_index(ref[column].to_numpy(), cur[column].to_numpy())),
            ks=ks_stat(ref[column].to_numpy(), cur[column].to_numpy()),
            drifted=psi >= psi_thresh,
        )
        for column in columns
    ]

    share = float(np.mean([r.drifted for r in results])) if results else 0.0

    novelties = {
        column: new_categories(ref[column], cur[column])
        for column in categorical_columns
        if column in ref.columns
    }

    return {
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "psi_threshold": psi_thresh,
        "share_threshold": share_thresh,
        "drift_share": share,
        "dataset_drift": bool(share >= share_thresh),
        "new_categories": {k: v for k, v in novelties.items() if v},
        "columns": [asdict(r) for r in results],
    }
