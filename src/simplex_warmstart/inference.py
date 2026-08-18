"""Prediction"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import FEATURE_COLS, PERMUTATION_INDEX, Standardizer
from .model import MixtureMLP


class SymmetricPredictor:
    """Mean les preds sur toutes les perm des composants"""

    def __init__(self, model: MixtureMLP, scaler: Standardizer, metadata: dict | None = None):
        self.model = model.eval()
        self.scaler = scaler
        self.metadata = metadata

    @classmethod
    def from_artifacts(cls, model_path: Path, metadata_path: Path) -> SymmetricPredictor:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = MixtureMLP(
            n_features=metadata["model"]["n_features"],
            hidden=metadata["model"]["hidden"],
            dropout=metadata["model"]["dropout"],
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return cls(model, Standardizer.from_dict(metadata.pop("scaler")), metadata)

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            outputs = [
                self.model(torch.from_numpy(self.scaler.transform(x[:, index]))).numpy()
                for index in PERMUTATION_INDEX
            ]
        return np.mean(outputs, axis=0)

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict(df[FEATURE_COLS].to_numpy(dtype=np.float32))
