import pandera.errors as pa_errors
import pytest

from simplex_warmstart.schemas import validate_experiments
from simplex_warmstart.simulate import generate_batch


@pytest.fixture
def batch():
    return generate_batch(0, n_studies=5, seed=0)


def test_generated_batch_validates_contract(batch):
    validate_experiments(batch)


def test_broken_composition_rejected(batch):
    batch.loc[0, "x1"] = batch.loc[0, "x1"] + 0.1  # somme > 1
    with pytest.raises(pa_errors.SchemaErrors):
        validate_experiments(batch)


def test_unknown_family_rejected(batch):
    batch.loc[0, "family"] = "inconnue"
    with pytest.raises(pa_errors.SchemaErrors):
        validate_experiments(batch)


def test_unexpected_column_rejected(batch):
    batch["colonne_inconnue"] = 1.0
    with pytest.raises(pa_errors.SchemaErrors):
        validate_experiments(batch)
