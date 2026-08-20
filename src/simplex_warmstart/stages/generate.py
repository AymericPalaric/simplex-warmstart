"""E1 : lot brut"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..params import load_params
from ..simulate import generate_batch

OUTPUT = Path("data/raw/experiments.parquet")


def main():
    cfg = load_params()["data"]
    batches = cfg["batches"]
    ids = [b["id"] for b in batches]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicates in batch ids : {ids}")

    frame = pd.concat(
        [
            generate_batch(
                batch_id=b["id"],
                n_studies=b["n_studies"],
                families=tuple(b["families"]),
                seed=b["seed"],
                protocol=b.get("protocol", "v1"),
            )
            for b in batches
        ],
        ignore_index=True,
    )

    if frame["study_id"].nunique() != sum(b["n_studies"] for b in batches):
        raise ValueError("Study ids duplicated between batches")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"{len(frame)} lignes, {frame['study_id'].nunique()} études -> {OUTPUT}")


if __name__ == "__main__":
    main()
