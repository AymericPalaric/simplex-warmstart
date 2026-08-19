"""Logique de promotion : challenger devient champion si meilleure perf sur golden set"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from simplex_warmstart.params import load_params
from simplex_warmstart.tracking import setup_mlflow

GOLDEN_PATH = Path("tests/golden/experiments.parquet")
REPORT_PATH = Path("metrics/promotion.md")


def score_alias(name: str, alias: str, golden: pd.DataFrame) -> float:
    model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
    pred = np.asarray(model.predict(golden)).ravel()
    target = golden["y"].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def resolve(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def main():
    cfg = load_params()["registry"]
    name, margin = cfg["model_name"], cfg["promotion_margin"]
    golden = pd.read_parquet(GOLDEN_PATH)

    setup_mlflow()
    client = MlflowClient()

    challenger = resolve(client, name, "challenger")
    if challenger is None:
        print("No challenger recorded")
        sys.exit(1)

    challenger_rmse = score_alias(name, "challenger", golden)
    client.set_model_version_tag(name, challenger.version, "golden_rmse", f"{challenger_rmse:.6f}")

    champion = resolve(client, name, "champion")
    if champion is None:
        decision, champion_rmse = True, None
        reason = "No active champion"
    else:
        champion_rmse = score_alias(name, "champion", golden)
        decision = challenger_rmse <= champion_rmse - margin
        reason = f"Perf gain of {champion_rmse - challenger_rmse:+.4f} (required margin : {margin})"

    lines = [
        "## Promotion decision",
        "",
        "| Version | Alias | RMSE golden |",
        "| --- | --- | --- |",
        f"| {challenger.version} | challenger | {challenger_rmse:.4f} |",
    ]

    if champion is not None:
        lines.append(f"| {champion.version} | champion | {champion_rmse:.4f} |")
    lines += ["", f"**{'PROMOTED' if decision else 'DENIED'}** — {reason}."]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if decision:
        if champion is not None:
            # Trace du prédécesseur pour rollback if needed
            client.set_model_version_tag(
                name, challenger.version, "fallen_champion", champion.version
            )
        client.set_registered_model_alias(name, "champion", challenger.version)


if __name__ == "__main__":
    main()
