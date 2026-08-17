"""Conf MLFlow"""

from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "simplex-warmstart"


def setup_mlflow():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)
