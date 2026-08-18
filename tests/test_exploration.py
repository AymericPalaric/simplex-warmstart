from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simplex_warmstart.chemistry import N_COMPONENTS, response
from simplex_warmstart.exploration import (
    build_frame,
    convex_hull,
    region_size,
    sample_simplex,
    suggest_region,
)
from simplex_warmstart.features import COMPOSITION_COLS, FEATURE_COLS

DESCRIPTORS = np.array([[0.1, 0.2], [0.0, -0.1], [0.3, 0.4]])


def chemistry_predict(frame: pd.DataFrame) -> np.ndarray:
    """Oracle : la vraie chimie, pour tester l'exploration sans modèle entraîné"""
    compositions = frame[COMPOSITION_COLS].to_numpy(dtype=float)
    return response(compositions, DESCRIPTORS)


def first_component_predict(frame: pd.DataFrame) -> np.ndarray:
    """Optimum trivial : le sommet x1 = 1"""
    return frame["x1"].to_numpy(dtype=float)


def is_inside(polygon: np.ndarray, point: np.ndarray) -> bool:
    """Point dans un polygone convexe orienté en sens trigonométrique"""
    edges = np.roll(polygon, -1, axis=0) - polygon
    to_point = point - polygon
    cross = edges[:, 0] * to_point[:, 1] - edges[:, 1] * to_point[:, 0]
    return bool((cross >= -1e-9).all())


def test_samples_stay_on_the_simplex():
    compositions = sample_simplex(500, seed=0)
    assert compositions.shape == (500, N_COMPONENTS)
    assert (compositions >= 0.0).all()
    assert np.allclose(compositions.sum(axis=1), 1.0)


def test_sampling_is_reproducible():
    assert np.array_equal(sample_simplex(64, seed=3), sample_simplex(64, seed=3))
    assert not np.array_equal(sample_simplex(64, seed=3), sample_simplex(64, seed=4))


def test_frame_matches_the_training_features():
    compositions = sample_simplex(8, seed=1)
    frame = build_frame(compositions, DESCRIPTORS)
    assert list(frame.columns) == FEATURE_COLS
    assert np.allclose(frame[COMPOSITION_COLS].to_numpy(), compositions)
    # Descripteurs constants, aplatis composant par composant comme à l'entraînement
    assert np.allclose(frame.iloc[0, N_COMPONENTS:].to_numpy(), DESCRIPTORS.reshape(-1))
    assert frame["c2_d1"].nunique() == 1


def test_hull_keeps_only_the_corners():
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    inside = np.array([[0.5, 0.5], [0.3, 0.7], [0.6, 0.2]])
    points = np.vstack([inside, corners])

    hull = convex_hull(points)
    assert sorted(hull) == [3, 4, 5, 6]

    # Sens trigonométrique => aire signée positive
    loop = points[hull]
    shifted = np.roll(loop, -1, axis=0)
    assert float(np.sum(loop[:, 0] * shifted[:, 1] - shifted[:, 0] * loop[:, 1])) > 0


def test_hull_of_collinear_points_is_a_segment():
    points = np.column_stack([np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5)])
    assert sorted(convex_hull(points)) == [0, 4]


def test_region_size_follows_the_fraction_with_a_floor():
    assert region_size(4000, 0.05) == 200
    assert region_size(100, 0.001) == 3  # jamais moins qu'un triangle
    assert region_size(5, 1.0) == 5


def test_region_is_the_top_slice_of_the_predictions():
    suggestion = suggest_region(chemistry_predict, DESCRIPTORS, n_samples=1000, seed=0)

    assert len(suggestion.top_index) == 50
    assert suggestion.best_prediction == suggestion.predictions.max()
    assert (suggestion.region_predictions >= suggestion.threshold).all()

    outside = np.setdiff1d(np.arange(1000), suggestion.top_index)
    assert suggestion.predictions[outside].max() <= suggestion.threshold


def test_envelope_wraps_every_point_of_the_region():
    suggestion = suggest_region(chemistry_predict, DESCRIPTORS, n_samples=2000, seed=1)

    assert len(suggestion.envelope_index) >= 3
    assert set(suggestion.envelope_index) <= set(suggestion.top_index)
    assert np.allclose(suggestion.envelope.sum(axis=1), 1.0)

    projected = suggestion.compositions @ np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    polygon = projected[suggestion.envelope_index]
    assert all(is_inside(polygon, projected[i]) for i in suggestion.top_index)


def test_centroid_and_bounds_describe_the_region():
    suggestion = suggest_region(chemistry_predict, DESCRIPTORS, n_samples=1000, seed=2)

    assert np.isclose(suggestion.centroid.sum(), 1.0)
    low, high = suggestion.bounds[:, 0], suggestion.bounds[:, 1]
    assert (low <= suggestion.centroid).all() and (suggestion.centroid <= high).all()
    assert (low <= suggestion.best_composition).all()
    assert (suggestion.best_composition <= high).all()
    # Une région, pas tout le simplexe
    assert (high - low).sum() < 3.0


def test_region_points_at_the_optimum():
    suggestion = suggest_region(first_component_predict, DESCRIPTORS, n_samples=2000, seed=0)
    assert suggestion.best_composition[0] > 0.9
    assert suggestion.centroid[0] > 0.8


def test_minimize_flips_the_region():
    kwargs = {"n_samples": 2000, "seed": 0}
    highest = suggest_region(first_component_predict, DESCRIPTORS, **kwargs)
    lowest = suggest_region(first_component_predict, DESCRIPTORS, objective="minimize", **kwargs)
    assert lowest.best_prediction < highest.best_prediction
    assert lowest.centroid[0] < 0.2
    assert not set(highest.top_index) & set(lowest.top_index)


def test_suggestion_is_deterministic():
    kwargs = {"n_samples": 500, "seed": 7}
    first = suggest_region(chemistry_predict, DESCRIPTORS, **kwargs)
    second = suggest_region(chemistry_predict, DESCRIPTORS, **kwargs)
    assert np.array_equal(first.envelope, second.envelope)
    assert first.best_prediction == second.best_prediction


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"objective": "whatever"}, "objective"),
        ({"n_samples": 0}, "n_samples"),
        ({"top_fraction": 0.0}, "top_fraction"),
        ({"top_fraction": 1.5}, "top_fraction"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        suggest_region(chemistry_predict, DESCRIPTORS, **kwargs)
