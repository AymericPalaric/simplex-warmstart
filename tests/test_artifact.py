from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from simplex_warmstart.features import FEATURE_COLS, PERMUTATION_INDEX
from simplex_warmstart.params import load_params

from .conftest import GOLDEN_PATH, METADATA_PATH

pytestmark = pytest.mark.artifact


def test_metadata_matches_current_feats():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["feature_cols"] == FEATURE_COLS


def test_gs_performance_holds(artifact_predictor):
    golden = pd.read_parquet(GOLDEN_PATH)
    pred = artifact_predictor.predict_df(golden)
    target = golden["y"].to_numpy(dtype=np.float32).reshape(-1, 1)
    value = float(np.sqrt(np.mean((pred - target) ** 2)))
    threshold = load_params()["gates"]["max_test_rmse"]
    assert value <= threshold, f"RMSE golden {value:.4f} > seuil {threshold}"


def test_artifact_is_perm_invariant(artifact_predictor):
    golden = pd.read_parquet(GOLDEN_PATH)
    x = golden[FEATURE_COLS].to_numpy(dtype=np.float32)
    reference = artifact_predictor.predict(x)
    for index in PERMUTATION_INDEX:
        assert np.allclose(artifact_predictor.predict(x[:, index]), reference, atol=1e-5)
