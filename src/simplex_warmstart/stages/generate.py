"""E1 : lot brut"""

from __future__ import annotations

from pathlib import Path

from ..params import load_params
from ..simulate import generate_batch

OUTPUT = Path("data/raw/experiments.parquet")


def main():
    cfg = load_params()["data"]
    frame = generate_batch(
        batch_id=cfg["batch_id"],
        n_studies=cfg["n_studies"],
        families=tuple(cfg["families"]),
        seed=cfg["seed"],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"{len(frame)} lignes, {frame['study_id'].nunique()} études -> {OUTPUT}")


if __name__ == "__main__":
    main()
