"""Golden Set - Ref inchangée"""

from pathlib import Path

from simplex_warmstart.simulate import generate_batch

OUTPUT = Path("tests/golden/experiments.parquet")

if __name__ == "__main__":
    frame = generate_batch(batch_id=99, n_studies=20, seed=20261808)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"{len(frame)} lignes -> {OUTPUT}")
