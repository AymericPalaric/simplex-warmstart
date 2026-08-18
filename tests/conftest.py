from pathlib import Path

import pytest

MODEL_PATH = Path("models/model.pt")
METADATA_PATH = Path("models/metadata.json")
GOLDEN_PATH = Path("tests/golden/experiments.parquet")


@pytest.fixture(scope="session")
def artifact_predictor():
    from simplex_warmstart.inference import SymmetricPredictor

    if not (MODEL_PATH.exists() and METADATA_PATH.exists()):
        pytest.skip("No artifact : run `dvc repro` first")
    return SymmetricPredictor.from_artifacts(MODEL_PATH, METADATA_PATH)
