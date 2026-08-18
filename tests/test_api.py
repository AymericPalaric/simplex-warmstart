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
