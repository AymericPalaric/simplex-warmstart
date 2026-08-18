import numpy as np
import torch

from simplex_warmstart.features import FEATURE_COLS, LEAKY_COLS, Standardizer, build_xy
from simplex_warmstart.model import MixtureMLP
from simplex_warmstart.simulate import generate_batch


def test_leakage_columns_excluded():
    assert not set(FEATURE_COLS) & set(LEAKY_COLS)


def test_build_xy_shapes():
    frame = generate_batch(0, n_studies=3, seed=0)
    x, y = build_xy(frame)
    assert x.shape == (len(frame), len(FEATURE_COLS))
    assert y.shape == (len(frame), 1)
    assert x.dtype == np.float32


def test_standardizer_with_constant_column():
    x = np.ones((10, 3), dtype=np.float32)
    x[:, 0] = np.arange(10)
    scaler = Standardizer.fit(x)
    transformed = scaler.transform(x)
    assert np.isfinite(transformed).all()


def test_standardizer_roundtrips_json():
    x = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    scaler = Standardizer.fit(x)
    restored = Standardizer.from_dict(scaler.to_dict())
    assert np.allclose(scaler.transform(x), restored.transform(x))


def test_model_output_shape():
    model = MixtureMLP(n_features=len(FEATURE_COLS), hidden=(8,))
    out = model(torch.zeros(5, len(FEATURE_COLS)))
    assert out.shape == (5, 1)
