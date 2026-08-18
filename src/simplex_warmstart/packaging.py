"""Packaging du modèle au format pyfunc : .pt, scaler|preproc, sequencer"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from .features import FEATURE_COLS

MODEL_PATH = Path("models/model.pt")
METADATA_PATH = Path("models/metadata.json")


class WarmstartModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from .inference import SymmetricPredictor

        self._predictor = SymmetricPredictor.from_artifacts(
            Path(context.artifacts["weights"]), Path(context.artifacts["metadata"])
        )

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        return self._predictor.predict_df(model_input).ravel()


def build_signature(sample: pd.DataFrame):
    from .inference import SymmetricPredictor

    predictor = SymmetricPredictor.from_artifacts(MODEL_PATH, METADATA_PATH)
    return mlflow.models.infer_signature(sample[FEATURE_COLS], predictor.predict_df(sample).ravel())


def log_pyfunc_model(sample: pd.DataFrame) -> str:
    """Log le modèle dans le run, avec son URI"""
    info = mlflow.pyfunc.log_model(
        name="warmstart",
        python_model=WarmstartModel(),
        artifacts={"weights": str(MODEL_PATH), "metadata": str(METADATA_PATH)},
        signature=build_signature(sample),
        input_example=sample[FEATURE_COLS].head(3),
        pip_requirements=["torch", "numpy", "pandas", "mlflow"],
    )
    return info.model_uri
