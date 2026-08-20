"""
Construit la Vérité terrain (= la chimie à retrouver)

Entrées = 3 composants, décrits par n descripteurs chacun
Réponse = polynôme de Scheffé quadratique dont coeff sont déterminés par les descripteurs
"""

from __future__ import annotations

import numpy as np

N_COMPONENTS = 3
N_DESCRIPTORS = 2

# Constantes de la chimie
_PURE_BASE = 5.0
_PURE_W = np.array([1.8, -0.9])
_SYNERGY = 2.4
_ANTAGONISM = 3.1
PROTOCOLS = {  # dérive de concept pure
    "v1": {"antagonism": 3.1, "offset": 0.0},
    "v2": {"antagonism": 5.4, "offset": -1.2},
}

PAIRS = ((0, 1), (0, 2), (1, 2))


def pure_coefficients(descriptors: np.ndarray) -> np.ndarray:
    """Coefficients b_i des termes purs. descriptors.shape = (N_COMPONENTS, N_DESCRIPTORS)
    -> b.shape = (N_COMPONENTS,)"""
    linear = descriptors @ _PURE_W
    return _PURE_BASE + linear + 0.4 * descriptors[:, 0] ** 2


def binary_coefficients(descriptors: np.ndarray, antagonism: float = _ANTAGONISM) -> np.ndarray:
    """Coefficients b_ij des termes croisés, dans l'ordre de PAIRS.
    descriptors.shape = (N_COMPONENTS, N_DESCRIPTORS) -> b.shape = (len(PAIRS),)"""
    out = np.empty(len(PAIRS))
    for k, (i, j) in enumerate(PAIRS):
        gap = descriptors[i] - descriptors[j]
        synergy = _SYNERGY * descriptors[i, 1] * descriptors[j, 1]
        out[k] = synergy - antagonism * float(gap @ gap)
    return out


def response(compositions: np.ndarray, descriptors: np.ndarray, protocol: str = "v1") -> np.ndarray:
    """Réponse sans bruit.
    compositions.shape = (n, N_COMPONENTS), descriptors.shape = (N_COMPONENTS, N_DESCRIPTORS)
    -> response.shape = (n,)"""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol : {protocol}")
    config = PROTOCOLS[protocol]

    pure = pure_coefficients(descriptors)
    binary = binary_coefficients(descriptors, antagonism=config["antagonism"])
    y = compositions @ pure
    for k, (i, j) in enumerate(PAIRS):
        y += binary[k] * compositions[:, i] * compositions[:, j]
    return y + config["offset"]
