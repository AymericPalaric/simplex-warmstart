"""E3 : train"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
import torch

from ..features import FEATURE_COLS, TARGET_COL, build_xy
from ..params import load_params
from ..tracking import setup_mlflow
from ..training import train_model

INPUT = Path("data/processed/experiments.parquet")
MODEL_PATH = Path("models/model.pt")
METADATA_PATH = Path("models/metadata.json")
METRICS_PATH = Path("metrics/train.json")


def main():
    params = load_params()
    cfg_model, cfg_train = params["model"], params["train"]
    torch.manual_seed(cfg_train["seed"])

    frame = pd.read_parquet(INPUT)
    train_df = frame[frame["split"] == "train"]
    val_df = frame[frame["split"] == "val"]
    x_val, y_val = build_xy(val_df)

    setup_mlflow()
    with mlflow.start_run() as run:
        mlflow.log_params({f"model.{k}": v for k, v in cfg_model.items()})
        mlflow.log_params({f"train.{k}": v for k, v in cfg_train.items()})
        mlflow.log_params(
            {
                "data.train_studies": int(train_df["study_id"].nunique()),
                "data.val_studies": int(val_df["study_id"].nunique()),
                "data.n_features": len(FEATURE_COLS),
            }
        )

        def log_epoch(epoch: int, metrics: dict[str, float]) -> None:
            mlflow.log_metrics(metrics, step=epoch)

        result = train_model(train_df, val_df, cfg_model, cfg_train, on_epoch_callback=log_epoch)

        mlflow.log_metric("best_val_rmse", result.best_val_rmse)
        mlflow.log_metric("best_epoch", result.best_epoch)
        mlflow.pytorch.log_model(result.model, name="model", input_example=x_val)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result.model.state_dict(), MODEL_PATH)

        metadata = {
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
            "scaler": result.scaler.to_dict(),
            "model": {
                "n_features": len(FEATURE_COLS),
                "hidden": list(cfg_model["hidden"]),
                "dropout": cfg_model["dropout"],
            },
            "mlflow_run_id": run.info.run_id,
            "created_at": datetime.now(UTC).isoformat(),
        }

        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(
            json.dumps({"best_val_rmse": result.best_val_rmse}, indent=2), encoding="utf-8"
        )

    print(f"val_rmse={result.best_val_rmse:.4f} run={metadata['mlflow_run_id']}")


if __name__ == "__main__":
    main()
