"""Drift-aware ML API — Day 1 skeleton.

Serves a trained ML model (Iris first, then Churn) as a FastAPI service,
watches its input data for drift with Evidently AI, and proposes a retrain
when drift passes a threshold. Day 1 is the skeleton only: /health works,
/predict and /drift-report answer 501 until the model trains (Day 2) and
reference data exists (Day 4). A retrain never swaps a model silently —
the new model is reviewed before it replaces the old one.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="drift-aware-ml-api", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "drift-aware-ml-api"}


@app.post("/predict")
def predict(payload: dict):
    return JSONResponse(
        status_code=501,
        content={
            "error": "model_not_trained",
            "message": "Iris model trains on Day 2 — no predictions yet.",
            "received": payload,
        },
    )


@app.get("/drift-report")
def drift_report():
    return JSONResponse(
        status_code=501,
        content={
            "error": "no_reference_data",
            "message": "Reference data is saved on Day 4 — no drift to report yet.",
        },
    )
