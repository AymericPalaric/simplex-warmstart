from __future__ import annotations

import numpy as np
import pytest

from simplex_warmstart.chemistry import response
from simplex_warmstart.features import FEATURE_COLS, PERMUTATION_INDEX
from simplex_warmstart.inference import SymmetricPredictor
from simplex_warmstart.simulate import generate_batch
from simplex_warmstart.splits import assign_splits
from simplex_warmstart.training import train_model

CFG_MODEL = {"hidden": [32, 32], "dropout": 0.0}
CFG_TRAIN = {
    "epochs": 40,
    "batch_size": 256,
    "lr": 0.005,
    "weight_decay": 0.0001,
    "seed": 0,
    "augment_permutations": True,
}


@pytest.fixture(scope="session")
def predictor() -> SymmetricPredictor:
    df = assign_splits(generate_batch(0, n_studies=30, seed=7), 0.2, 0.2, seed=3)
    result = train_model(
        df[df["split"] == "train"],
        df[df["split"] == "val"],
        CFG_MODEL,
        CFG_TRAIN,
    )
    return SymmetricPredictor(result.model, result.scaler)


def random_inputs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    compositions = rng.dirichlet(np.ones(3), size=n)
    descriptors = rng.normal(0.0, 0.6, size=(n, 6))
    return np.hstack([compositions, descriptors]).astype(np.float32)


def test_gt_is_perm_invariant():
    x = random_inputs(64, seed=1)
    reference = np.array([response(row[:3].reshape(1, 3), row[3:].reshape(3, 2))[0] for row in x])

    for index in PERMUTATION_INDEX:
        permuted = x[:, index]
        values = np.array([response(r[:3].reshape(1, 3), r[3:].reshape(3, 2))[0] for r in permuted])
        assert np.allclose(values, reference, atol=1e-6)


def test_pred_is_invariant_to_comp_perm(predictor):
    x = random_inputs(64, seed=2)
    reference = predictor.predict(x)
    for index in PERMUTATION_INDEX:
        assert np.allclose(predictor.predict(x[:, index]), reference, atol=1e-5)


def test_pred_deterministic(predictor):
    x = random_inputs(32, seed=3)
    assert np.array_equal(predictor.predict(x), predictor.predict(x))


def test_pred_continuous(predictor):
    x = random_inputs(32, seed=4)
    nudged = x.copy()
    nudged[:, :3] += 1e-6
    nudged[:, :3] /= nudged[:, :3].sum(axis=1, keepdims=True)
    delta = np.abs(predictor.predict(nudged) - predictor.predict(x))
    assert delta.max() < 1e-2


def test_pred_plausible_range(predictor):
    x = random_inputs(256, seed=5)
    pred = predictor.predict(x)
    assert np.isfinite(pred).all()
    assert pred.min() > -20.0
    assert pred.max() < 30.0


def test_raise_first_desc_raise_pred(predictor):
    rng = np.random.default_rng(6)
    base = np.zeros((32, len(FEATURE_COLS)), dtype=np.float32)
    base[:, 0] = 1.0  # composition pure en composant 1
    base[:, 3:] = rng.normal(0.0, 0.4, size=(32, 6))

    low, high = base.copy(), base.copy()
    low[:, 3] = 0.0
    high[:, 3] = 1.0

    gain = predictor.predict(high) - predictor.predict(low)
    assert gain.mean() > 0.5
