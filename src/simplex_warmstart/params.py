"""Chargement des paramètres (vérité = params.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path("params.yaml")


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
