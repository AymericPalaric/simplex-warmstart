"""Evalue le model prod sur le batch le plus récent, sans retrain
-> Détecte la dérive de concept"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simplex_warmstart.inference import SymmetricPredictor

DATA_PATH = Path("data/processed/experiments.parquet")
OUTPUT_PATH = Path("metrics/champion_monitor.json")


def rmse(predictor: SymmetricPredictor, df: pd.DataFrame) -> float:
    pred = predictor.predict_df(df).ravel()
    target = df["y"].to_numpy(dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    predictor = SymmetricPredictor.from_artifacts(
        args.model_dir / "model.pt", args.model_dir / "metadata.json"
    )
    df = pd.read_parquet(DATA_PATH)
    target_batch = args.batch if args.batch is not None else int(df["batch_id"].max())

    ref = df[df["batch_id"] < target_batch]
    cur = df[df["batch_id"] == target_batch]
    if ref.empty or cur.empty:
        raise SystemExit("There must be at least 2 batches to compare")

    ref_rmse = rmse(predictor, ref)
    cur_rmse = rmse(predictor, cur)

    payload = {
        "model_dir": str(args.model_dir),
        "batch": target_batch,
        "rmse_reference": ref_rmse,
        "rmse_current": cur_rmse,
        "degradation_ratio": cur_rmse / ref_rmse,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Production model - Legacy RMSE {ref_rmse:.4f} | "
        f"Batch {target_batch} {cur_rmse:.4f} "
        f"(x{payload['degradation_ratio']:.2f})"
    )


if __name__ == "__main__":
    main()
