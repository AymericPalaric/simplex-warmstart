"""Load le modèle servi
Registry en prod, artefacts en dev"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features import FEATURE_COLS

DEFAULT_MODEL_URI = "models:/simplex-warmstart@champion"


@dataclass
class LoadedModel:
    predict_fn: Any
    source: str
    version: str | None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.predict_fn(df[FEATURE_COLS])).ravel()


def load_model(uri: str | None = None) -> LoadedModel:
    uri = uri or os.environ.get("MODEL_URI", DEFAULT_MODEL_URI)

    if uri == "local":
        from ..inference import SymmetricPredictor

        predictor = SymmetricPredictor.from_artifacts(
            Path("models/model.pt"), Path("models/metadata.json")
        )
        return LoadedModel(lambda df: predictor.predict_df(df), source="local", version=None)

    import mlflow
    from mlflow.tracking import MlflowClient

    from ..tracking import setup_mlflow

    setup_mlflow()
    model = mlflow.pyfunc.load_model(uri)
    version = None
    if "@" in uri:
        name, alias = uri.removeprefix("models:/").split("@")
        version = MlflowClient().get_model_version_by_alias(name, alias).version
    return LoadedModel(model.predict, source=uri, version=version)
