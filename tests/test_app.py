"""Day 1 skeleton tests: /health works, /predict and /drift-report
honestly answer 501 until the model and reference data exist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml-api"))

from fastapi.testclient import TestClient

from app import app


def _client():
    return TestClient(app)


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "service": "drift-aware-ml-api"}


def test_predict_not_trained_yet():
    resp = _client().post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert resp.status_code == 501
    assert resp.json()["error"] == "model_not_trained"


def test_drift_report_no_reference_data_yet():
    resp = _client().get("/drift-report")
    assert resp.status_code == 501
    assert resp.json()["error"] == "no_reference_data"
