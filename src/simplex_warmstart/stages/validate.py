"""E2 : contrat+split"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..params import load_params
from ..schemas import validate_experiments
from ..splits import assign_splits

INPUT = Path("data/raw/experiments.parquet")
OUTPUT = Path("data/processed/experiments.parquet")


def main():
    cfg = load_params()["split"]
    frame = validate_experiments(pd.read_parquet(INPUT))
    frame = assign_splits(
        frame,
        val_frac=cfg["val_fraction"],
        test_frac=cfg["test_fraction"],
        seed=cfg["seed"],
    )

    overlap = frame.groupby("study_id")["split"].nunique().max()
    if overlap != 1:
        raise AssertionError("Splits are overlapping : some studies are in multiple splits")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    counts = frame.groupby("split")["study_id"].nunique().to_dict()
    print(f"études par split : {counts} -> {OUTPUT}")


if __name__ == "__main__":
    main()
