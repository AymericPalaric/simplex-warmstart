"""Découpe des données en train/validation/test pour le ML
Découpage par étude"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_splits(
    frame: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> pd.DataFrame:
    """Découpe les études en train/val/test
    Ajoute une colonne 'split' au df d'expériences"""
    studies = np.sort(frame["study_id"].unique())
    rng = np.random.default_rng(seed)
    studies = studies[rng.permutation(len(studies))]  # shuffle

    total = len(studies)
    n_val = max(1, round(total * val_frac))
    n_test = max(1, round(total * test_frac))
    if n_val + n_test >= total:
        raise ValueError(f"val_frac + test_frac too large: {val_frac + test_frac} >= 1.0")

    mapping = {}
    mapping.update({s: "val" for s in studies[:n_val]})
    mapping.update({s: "test" for s in studies[n_val : n_val + n_test]})
    mapping.update({s: "train" for s in studies[n_val + n_test :]})

    out = frame.copy()
    out["split"] = out["study_id"].map(mapping)
    return out
