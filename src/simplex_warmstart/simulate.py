"""
Simule des expériences que ferait un laboratoire : produit des études

Etudes contraintes par une "famille" chimique =
région de l'espace des descripteurs pour tirer des composants
--> permet le OOD (dérive)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import N_COMPONENTS, N_DESCRIPTORS, response


@dataclass(frozen=True)
class Family:
    """Famille chimique = région de l'espace des descripteurs pour tirer des composants"""

    name: str
    center: tuple[float, ...]  # shape = (N_DESCRIPTORS,)
    radius: float


FAMILIES: dict[str, Family] = {
    "esters": Family(name="esters", center=(0.0, 0.0), radius=0.35),
    "silicones": Family(name="silicones", center=(-1.4, -0.6), radius=0.35),
    "alcanes": Family(name="alcanes", center=(-1.1, 1.2), radius=0.35),
}

BASELINE_FAMILIES = ("esters", "silicones")


def simplex_lattice(degree: int) -> np.ndarray:
    """Réseau Scheffé degré m : (m+1)(m+2)/2 points"""
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    points = [
        (i / degree, j / degree, (degree - i - j) / degree)
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ]
    return np.array(sorted(points))


def generate_study(
    rng: np.random.Generator,
    family: Family,
    study_id: str,
    batch_id: int,
    degree: int = 3,
    n_random: int = 4,
    noise_std: float = 0.15,
) -> pd.DataFrame:
    """Une étude = composants d'une famille + plan + mesures"""
    descriptors = rng.normal(family.center, family.radius, size=(N_COMPONENTS, N_DESCRIPTORS))
    compositions = np.vstack(
        [simplex_lattice(degree), rng.dirichlet(np.ones(N_COMPONENTS), size=n_random)]
    )

    y_true = response(compositions, descriptors)
    y = y_true + rng.normal(0, noise_std, size=len(compositions))

    df = pd.DataFrame(compositions, columns=[f"x{i + 1}" for i in range(N_COMPONENTS)])
    for c in range(N_COMPONENTS):
        for d in range(N_DESCRIPTORS):
            df[f"c{c + 1}_d{d + 1}"] = descriptors[c, d]

    df["y_true"] = y_true
    df["y"] = y

    df.insert(0, "study_id", study_id)
    df.insert(0, "family", family.name)
    df.insert(0, "batch_id", batch_id)
    return df


def generate_batch(
    batch_id: int,
    n_studies: int = 40,
    families: tuple[str, ...] = BASELINE_FAMILIES,
    seed: int = 0,
) -> pd.DataFrame:
    """Un batch = les études arrivées sur une période de temps"""
    rng = np.random.default_rng(seed)
    frames = [
        generate_study(
            rng=rng,
            family=FAMILIES[families[k % len(families)]],
            study_id=f"batch_{batch_id:02d}-study_{k:03d}",
            batch_id=batch_id,
        )
        for k in range(n_studies)
    ]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Génère un batch d'études")
    parser.add_argument("--batch_id", "-b", type=int, required=True, help="Taille du batch")
    parser.add_argument("--n_studies", "-s", type=int, default=40, help="Nombre d'études à générer")
    parser.add_argument(
        "--families",
        "-f",
        type=str,
        nargs="+",
        default=list(BASELINE_FAMILIES),
        help="Familles chimiques à inclure",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reproductibilité")
    parser.add_argument("--out", type=Path, required=True, help="Fichier de sortie parquet")
    args = parser.parse_args()

    frame = generate_batch(
        batch_id=args.batch_id,
        n_studies=args.n_studies,
        families=tuple(args.families),
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    print(f"{len(frame)} lignes, {frame.study_id.nunique()} études -> {args.out}")


if __name__ == "__main__":
    main()
