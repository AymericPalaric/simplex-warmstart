"""E4 : test"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from ..features import Standardizer, build_xy
from ..model import MixtureMLP
from ..tracking import setup_mlflow

INPUT = Path("data/processed/experiments.parquet")
MODEL_PATH = Path("models/model.pt")
METADATA_PATH = Path("models/metadata.json")
METRICS_PATH = Path("metrics/eval.json")


def regression_scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = pred - target
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def main():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    scaler = Standardizer.from_dict(metadata["scaler"])

    model = MixtureMLP(
        n_features=metadata["model"]["n_features"],
        hidden=metadata["model"]["hidden"],
        dropout=metadata["model"]["dropout"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    frame = pd.read_parquet(INPUT)
    test_df = frame[frame["split"] == "test"]
    x_test, y_test = build_xy(test_df)

    with torch.no_grad():
        pred = model(torch.from_numpy(scaler.transform(x_test))).numpy()

    results = {"test": regression_scores(pred, y_test)}
    for family, index in test_df.groupby("family").indices.items():
        results[f"test_{family}"] = regression_scores(pred[index], y_test[index])

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    setup_mlflow()
    with mlflow.start_run(run_id=metadata["mlflow_run_id"]):
        mlflow.log_metrics(
            {f"{scope}.{k}": v for scope, scores in results.items() for k, v in scores.items()}
        )

    print(json.dumps(results["test"], indent=2))


if __name__ == "__main__":
    main()
