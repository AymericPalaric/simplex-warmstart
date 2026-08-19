"""Save le modèle courant en tant que challenger - si la quality gate est passée"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from simplex_warmstart.packaging import METADATA_PATH, log_pyfunc_model
from simplex_warmstart.params import load_params
from simplex_warmstart.tracking import setup_mlflow

GATE_PATH = Path("metrics/gate.json")
GOLDEN_PATH = Path("tests/golden/experiments.parquet")


def main():
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    if not gate["passed"]:
        print("Quality gate failed : no record.")
        sys.exit(1)

    cfg = load_params()["registry"]
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    sample = pd.read_parquet(GOLDEN_PATH)

    setup_mlflow()
    with mlflow.start_run(run_id=metadata["mlflow_run_id"]):
        model_uri = log_pyfunc_model(sample)

    version = mlflow.register_model(model_uri, cfg["model_name"]).version
    client = MlflowClient()
    client.set_registered_model_alias(cfg["model_name"], "challenger", version)
    client.set_model_version_tag(cfg["model_name"], version, "run_id", metadata["mlflow_run_id"])
    print(f"Version {version} saved with alias @challenger")


if __name__ == "__main__":
    main()
