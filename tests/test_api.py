from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from simplex_warmstart.serving import app as app_module

pytestmark = pytest.mark.artifact

VALID = {
    "formulations": [
        {"composition": [0.5, 0.3, 0.2], "descriptors": [[0.1, 0.2], [0.0, -0.1], [0.3, 0.4]]}
    ]
}


@pytest.fixture(scope="module")
def client(monkeypatch_session=None):
    import os

    os.environ["MODEL_URI"] = "local"
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health_reports_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_predict_returns_one_value_per_formulation(client):
    body = client.post("/predict", json=VALID).json()
    assert len(body["predictions"]) == 1
    assert body["model_source"] == "local"


def test_composition_not_summing_to_one_is_rejected(client):
    payload = {"formulations": [{"composition": [0.5, 0.3, 0.9], "descriptors": [[0, 0]] * 3}]}
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_fraction_is_rejected(client):
    payload = {"formulations": [{"composition": [1.2, -0.2, 0.0], "descriptors": [[0, 0]] * 3}]}
    assert client.post("/predict", json=payload).status_code == 422


def test_permutation_invariance_holds_through_the_api(client):
    direct = client.post("/predict", json=VALID).json()["predictions"][0]
    swapped = {
        "formulations": [
            {
                "composition": [0.3, 0.5, 0.2],
                "descriptors": [[0.0, -0.1], [0.1, 0.2], [0.3, 0.4]],
            }
        ]
    }
    assert client.post("/predict", json=swapped).json()["predictions"][0] == pytest.approx(
        direct, abs=1e-4
    )


REGION = {"descriptors": [[0.1, 0.2], [0.0, -0.1], [0.3, 0.4]], "n_samples": 1000, "seed": 0}


def test_suggest_region_returns_a_polygon_on_the_simplex(client):
    body = client.post("/suggest-region", json=REGION).json()

    assert body["region_size"] == 50
    assert body["n_samples"] == 1000
    assert len(body["envelope"]) >= 3
    for point in [*body["envelope"], body["centroid"], body["best"]["composition"]]:
        assert len(point) == 3
        assert min(point) >= 0.0
        assert sum(point) == pytest.approx(1.0, abs=1e-6)


def test_suggest_region_reports_a_consistent_region(client):
    body = client.post("/suggest-region", json=REGION).json()

    assert body["best"]["prediction"] == pytest.approx(body["region_summary"]["max"])
    assert body["threshold"] == pytest.approx(body["region_summary"]["min"])
    assert body["region_summary"]["min"] <= body["region_summary"]["mean"]
    assert body["region_summary"]["mean"] <= body["region_summary"]["max"]

    for column, value in zip(["x1", "x2", "x3"], body["best"]["composition"], strict=True):
        assert body["bounds"][column]["min"] <= value <= body["bounds"][column]["max"]


def test_suggest_region_agrees_with_predict(client):
    body = client.post("/suggest-region", json=REGION).json()
    payload = {
        "formulations": [
            {"composition": body["best"]["composition"], "descriptors": REGION["descriptors"]}
        ]
    }
    direct = client.post("/predict", json=payload).json()["predictions"][0]
    assert direct == pytest.approx(body["best"]["prediction"], abs=1e-4)


def test_minimizing_returns_a_worse_optimum(client):
    highest = client.post("/suggest-region", json=REGION).json()
    lowest = client.post("/suggest-region", json={**REGION, "objective": "minimize"}).json()
    assert lowest["best"]["prediction"] < highest["best"]["prediction"]


def test_suggest_region_is_reproducible(client):
    first = client.post("/suggest-region", json=REGION).json()
    second = client.post("/suggest-region", json=REGION).json()
    assert first == second
    other_seed = client.post("/suggest-region", json={**REGION, "seed": 1}).json()
    assert other_seed["envelope"] != first["envelope"]


def test_suggest_region_rejects_malformed_requests(client):
    payloads = [
        {"descriptors": [[0.1, 0.2], [0.0, -0.1]]},  # 2 composants
        {"descriptors": [[0.1], [0.0, -0.1], [0.3, 0.4]]},  # descripteurs incomplets
        {**REGION, "objective": "whatever"},
        {**REGION, "n_samples": 10},  # sous le plancher d'échantillonnage
        {**REGION, "top_fraction": 0.0},
    ]
    for payload in payloads:
        assert client.post("/suggest-region", json=payload).status_code == 422
