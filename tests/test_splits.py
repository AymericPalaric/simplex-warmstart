import pytest

from simplex_warmstart.simulate import generate_batch
from simplex_warmstart.splits import assign_splits


@pytest.fixture
def batch():
    return generate_batch(0, n_studies=40, seed=0)


def test_no_overlapping_splits(batch):
    frame = assign_splits(batch, 0.15, 0.15, seed=17)
    assert frame.groupby("study_id")["split"].nunique().max() == 1


def test_every_row_is_assigned(batch):
    frame = assign_splits(batch, 0.15, 0.15, seed=17)
    assert frame["split"].isin({"train", "val", "test"}).all()


def test_split_deterministic(batch):
    a = assign_splits(batch, 0.15, 0.15, seed=17)
    b = assign_splits(batch, 0.15, 0.15, seed=17)
    assert a["split"].equals(b["split"])


def test_excessive_fractions_refused(batch):
    with pytest.raises(ValueError):
        assign_splits(batch, 0.6, 0.6, seed=17)
