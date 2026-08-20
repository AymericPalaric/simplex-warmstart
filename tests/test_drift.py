import numpy as np
import pandas as pd

from simplex_warmstart.drift import (
    compute_drift,
    ks_stat,
    new_categories,
    population_stability_index,
)

RNG = np.random.default_rng(0)


def test_psi_is_near_zero_for_identical_distributions():
    sample = RNG.normal(size=5000)
    other = RNG.normal(size=5000)
    assert population_stability_index(sample, other) < 0.1


def test_psi_grows_with_the_shift():
    reference = RNG.normal(size=5000)
    small = population_stability_index(reference, RNG.normal(0.3, 1.0, size=5000))
    large = population_stability_index(reference, RNG.normal(2.0, 1.0, size=5000))
    assert small < large
    assert large > 0.25


def test_ks_is_bounded_and_ordered():
    reference = RNG.normal(size=2000)
    identical = ks_stat(reference, RNG.normal(size=2000))
    shifted = ks_stat(reference, RNG.normal(3.0, 1.0, size=2000))
    assert 0.0 <= identical <= 1.0
    assert shifted > identical


def test_constant_column_does_not_crash():
    constant = np.ones(100)
    assert population_stability_index(constant, constant) == 0.0


def test_new_category_is_reported():
    reference = pd.Series(["esters", "silicones"])
    current = pd.Series(["esters", "alcanes"])
    assert new_categories(reference, current) == ["alcanes"]


def test_dataset_drift_fires_when_enough_columns_move():
    reference = pd.DataFrame({"a": RNG.normal(size=2000), "b": RNG.normal(size=2000)})
    current = pd.DataFrame(
        {"a": RNG.normal(3.0, 1.0, size=2000), "b": RNG.normal(3.0, 1.0, size=2000)}
    )
    result = compute_drift(reference, current, ["a", "b"], categorical_columns=())
    assert result["dataset_drift"] is True
    assert result["drift_share"] == 1.0
