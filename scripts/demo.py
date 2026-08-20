"""Rejoue l'ajout de drift : 2 dérives, detection et résolution"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PARAMS = Path("params.yaml")
BACKUP = Path(".params.yaml.demo-backup")
DEPLOYED = Path("demo/deployed")


def banner(title: str):
    print(f"\n\033[1;36m{'=' * 72}\n  {title}\n{'=' * 72}\033[0m\n")


def resolve(*command: str) -> list[str]:
    """Ancre les sous-processus sur l'interpréteur courant.

    Sous Windows, CreateProcess ne résout pas « python » / « dvc » via le PATH
    posé par `uv run` : on retomberait sur le Python système, sans dvc ni
    simplex_warmstart installés.
    """
    head, *rest = command
    if head == "python":
        return [sys.executable, *rest]
    if head == "dvc":
        return [sys.executable, "-m", "dvc", *rest]
    return list(command)


def run(*command: str):
    print(f"\033[2m$ {' '.join(command)}\033[0m")
    subprocess.run(resolve(*command), check=False)


def show(path: Path, keys: list[str]):
    if not path.exists():
        print(f"    (missing report : {path})")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        print(f"    {key:>18} : {payload.get(key)}")


def rewind():
    """Ramène params.yaml au socle d'entraînement du champion.

    Le stage `drift` compare le dernier batch à TOUS les précédents. Sans
    rembobinage, le lot alcanes de l'acte 1 entre dans la référence de l'acte 2,
    qui mesure alors sa disparition au lieu du changement de protocole.
    Chaque acte repart donc du même socle, celui sur lequel tourne la prod.
    """
    shutil.copy(BACKUP, PARAMS)


# Seeds figées : chaque lot est identique d'un acte à l'autre.
# 120 études et non 40 : en dessous, PSI est trop bruité pour que l'acte 2
# puisse conclure (29/30 faux positifs mesurés à 40 études, seuil 0.2).
ALCANES = ("--n-studies", "120", "--families", "alcanes", "--seed", "2001")
PROTOCOL_V2 = (
    "--n-studies",
    "120",
    "--families",
    "esters",
    "silicones",
    "--protocol",
    "v2",
    "--seed",
    "3001",
)


def main() -> None:
    shutil.copy(PARAMS, BACKUP)
    print("--- Début de la simualation de prod ---\n")
    try:
        banner("ACTE 0 — Situation nominale : on entraîne et on déploie")
        run("dvc", "repro")
        DEPLOYED.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "metadata.json"):
            shutil.copy(Path("models") / name, DEPLOYED / name)
        show(Path("metrics/eval.json"), ["test"])
        print("  Modèle figé dans demo/deployed : c'est notre « production ».")

        banner("ACTE 1 — Dérive de covariables : la famille « alcanes » arrive")
        rewind()
        run("python", "scripts/append_batch.py", *ALCANES)
        run("dvc", "repro", "validate", "drift")
        show(Path("metrics/drift.json"), ["dataset_drift", "drift_share", "new_categories"])
        print("\n  Le détecteur voit la dérive AVANT toute étiquette. Effet sur la production :")
        run("python", "scripts/monitor_champion.py", "--model-dir", str(DEPLOYED))

        banner("ACTE 2 — Dérive de concept : nouveau protocole de mesure")
        print("  On rembobine : le lot alcanes est retiré, le protocole v2 arrive")
        print("  sur le même socle. Les deux actes se comparent à la même référence.\n")
        rewind()
        run("python", "scripts/append_batch.py", *PROTOCOL_V2)
        run("dvc", "repro", "validate", "drift")
        show(Path("metrics/drift.json"), ["dataset_drift", "drift_share"])
        print("\n  Les distributions d'entrée n'ont PAS bougé — le détecteur ne voit rien.")
        print("  Seule la performance du modèle déployé révèle le problème :")
        run("python", "scripts/monitor_champion.py", "--model-dir", str(DEPLOYED))

        banner("ACTE 3 — Remédiation : ré-entraînement sur l'historique complet")
        rewind()
        # run("python", "scripts/append_batch.py", *ALCANES)
        run("python", "scripts/append_batch.py", *PROTOCOL_V2)
        run("dvc", "repro")
        show(Path("metrics/eval.json"), ["test"])
        show(Path("metrics/gate.json"), ["passed"])
        run("python", "scripts/monitor_champion.py", "--model-dir", "models")
        print("\n  Le nouveau modèle absorbe les deux lots. Le gate tranche.")

    finally:
        banner("Restauration de l'état initial")
        shutil.move(BACKUP, PARAMS)
        shutil.rmtree(DEPLOYED.parent, ignore_errors=True)
        run("dvc", "repro")
        print("params.yaml restauré, pipeline reconstruit.")
    print("\n--- Fin de la simulation de prod ---")


if __name__ == "__main__":
    sys.exit(main())
