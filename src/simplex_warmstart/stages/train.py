"""E3 : train"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..features import FEATURE_COLS, TARGET_COL, Standardizer, build_xy
from ..model import MixtureMLP
from ..params import load_params
from ..tracking import setup_mlflow

INPUT = Path("data/processed/experiments.parquet")
MODEL_PATH = Path("models/model.pt")
METADATA_PATH = Path("models/metadata.json")
METRICS_PATH = Path("metrics/train.json")


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def main():
    params = load_params()
    cfg_model, cfg_train = params["model"], params["train"]
    torch.manual_seed(cfg_train["seed"])

    frame = pd.read_parquet(INPUT)
    train_df = frame[frame["split"] == "train"]
    val_df = frame[frame["split"] == "val"]

    x_train, y_train = build_xy(train_df)
    x_val, y_val = build_xy(val_df)

    scaler = Standardizer.fit(x_train)
    x_train_t = torch.from_numpy(scaler.transform(x_train))
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(scaler.transform(x_val))

    model = MixtureMLP(
        n_features=len(FEATURE_COLS),
        hidden=cfg_model["hidden"],
        dropout=cfg_model["dropout"],
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"]
    )
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg_train["seed"]),
    )

    setup_mlflow()
    best_val = float("inf")
    best_state = {}

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

        for epoch in range(cfg_train["epochs"]):
            model.train()
            for xb, yb in train_loader:
                optimiser.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimiser.step()

            model.eval()
            with torch.no_grad():
                train_rmse = rmse(model(x_train_t).numpy(), y_train)
                val_rmse = rmse(model(x_val_t).numpy(), y_val)
            mlflow.log_metrics(
                {"train_rmse": train_rmse, "val_rmse": val_rmse, "train_loss": loss}, step=epoch
            )

            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        mlflow.log_metric("best_val_rmse", best_val)
        mlflow.pytorch.log_model(model, name="model", input_example=x_val)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, MODEL_PATH)

        metadata = {
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
            "scaler": scaler.to_dict(),
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
        METRICS_PATH.write_text(json.dumps({"best_val_rmse": best_val}, indent=2), encoding="utf-8")

    print(f"val_rmse={best_val:.4f} run={metadata['mlflow_run_id']}")


if __name__ == "__main__":
    main()
