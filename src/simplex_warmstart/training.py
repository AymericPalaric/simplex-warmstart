"""Training core"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import (
    FEATURE_COLS,
    SYMMETRY_GROUPS,
    Standardizer,
    augment_permutations,
    build_xy,
)
from .model import MixtureMLP


@dataclass
class TrainingResult:
    model: MixtureMLP
    scaler: Standardizer
    best_val_rmse: float
    best_epoch: int


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg_model: dict,
    cfg_train: dict,
    on_epoch_callback: Callable[[int, dict[str, float]], None] | None = None,  # hook post epoch
) -> TrainingResult:
    torch.manual_seed(cfg_train["seed"])

    x_train, y_train = build_xy(train_df)
    x_val, y_val = build_xy(val_df)

    if cfg_train.get("augment_permutations", True):
        x_train, y_train = augment_permutations(x_train, y_train)

    scaler = Standardizer.fit(x_train, groups=SYMMETRY_GROUPS)
    x_train_t = torch.from_numpy(scaler.transform(x_train))
    x_val_t = torch.from_numpy(scaler.transform(x_val))
    y_train_t = torch.from_numpy(y_train)

    model = MixtureMLP(
        n_features=len(FEATURE_COLS),
        hidden=cfg_model["hidden"],
        dropout=cfg_model["dropout"],
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"]
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=cfg_train["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg_train["seed"]),
    )

    best_val, best_epoch = float("inf"), -1
    best_state = {}

    for epoch in range(cfg_train["epochs"]):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            metrics = {
                "train_rmse": rmse(model(x_train_t).numpy(), y_train),
                "val_rmse": rmse(model(x_val_t).numpy(), y_val),
            }

        if on_epoch_callback is not None:
            on_epoch_callback(epoch, metrics)

        if metrics["val_rmse"] < best_val:
            best_val, best_epoch = metrics["val_rmse"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return TrainingResult(model, scaler, best_val, best_epoch)
