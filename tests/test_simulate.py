import numpy as np
import pytest

from simplex_warmstart.chemistry import N_COMPONENTS
from simplex_warmstart.simulate import generate_batch, simplex_lattice


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_lattice_has_expected_size(degree):
    points = simplex_lattice(degree)
    assert len(points) == (degree + 1) * (degree + 2) // 2
    assert points.shape[1] == N_COMPONENTS


def test_lattice_points_are_compositions():
    points = simplex_lattice(4)
    assert np.allclose(points.sum(axis=1), 1.0)
    assert (points >= 0).all()


def test_batch_is_reproducible():
    a = generate_batch(0, n_studies=5, seed=42)
    b = generate_batch(0, n_studies=5, seed=42)
    assert a.equals(b)


def test_seed_changes_batch():
    a = generate_batch(0, n_studies=5, seed=42)
    b = generate_batch(0, n_studies=5, seed=43)
    assert not a["y"].equals(b["y"])


def test_measurments_noise_match_spec():
    frame = generate_batch(0, n_studies=200, seed=0)
    residual = frame["y"] - frame["y_true"]
    assert residual.std() == pytest.approx(0.15, rel=0.1)
    assert residual.mean() == pytest.approx(0.0, abs=0.02)


def test_descriptors_are_constant_per_study():
    frame = generate_batch(0, n_studies=10, seed=1)
    desc_cols = [c for c in frame.columns if c.startswith("c")]
    spread = frame.groupby("study_id")[desc_cols].nunique()
    assert (spread == 1).all().all()
