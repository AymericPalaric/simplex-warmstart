"""Manifeste de batch"""

from __future__ import annotations

from .chemistry import PROTOCOLS
from .simulate import FAMILIES

SEED_OFFSET = 1000


def next_batch_entry(
    batches: list[dict],
    n_studies: int,
    families: list[str],
    seed: int | None = None,
    protocol: str = "v1",
) -> dict:
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families : {unknown}")
    if n_studies < 1:
        raise ValueError("`n_studies` must be positive")
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol : {protocol}")

    batch_id = max((b["id"] for b in batches), default=-1) + 1
    return {
        "id": batch_id,
        "n_studies": n_studies,
        "seed": SEED_OFFSET + batch_id if seed is None else seed,
        "families": list(families),
        "protocol": protocol,
    }


def summarise(batches: list[dict]) -> dict:
    families = sorted({f for b in batches for f in b["families"]})
    return {
        "n_batches": len(batches),
        "n_studies": sum(b["n_studies"] for b in batches),
        "families": families,
    }
