# Drift-Aware ML API

**Innovation: Drift Detection + Auto-Retraining**

Serve a trained ML model (Iris first, then Churn) as a FastAPI service,
watch its input data for drift with Evidently AI, and auto-trigger a
retrain when drift passes a threshold — as a *proposal*, never a silent
model swap. The new model is reviewed before it replaces the old one.

## Purpose

Turn the Project 1 Flask app's habits (Docker, CI, guardian) into an ML
service: a trained model behind an API, with its own health, prediction
logging, drift reporting and a retrain workflow.

## Architecture

See [docs/architecture.md](docs/architecture.md) — request → FastAPI
(`/predict`) → model → logged predictions → Evidently drift report
(`/drift-report`) → retrain proposal via GitHub Actions.

## Tools used

Python 3.12, FastAPI, scikit-learn, Evidently AI, pytest, ruff,
GitHub Actions. No paid hosting — local + Docker first, Hugging Face
Spaces or Render free tier for a live demo link.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ml-api/requirements.txt -r requirements-dev.txt
python -m pytest -q
uvicorn app:app --reload --app-dir ml-api
```

## Status (Day 1)

Skeleton only: `/health` works. `/predict` answers 501 until the Iris
model trains (Day 2); `/drift-report` answers 501 until reference data
exists (Day 4).

## Measurable impact

- Drift detection lead time: days from drift injection to detection,
  measured with a deliberately shifted test set (Day 5).
- Retrain time saved vs manual retraining (timed once, Day 6 — not
  claimed in advance).

## What I learned

- _To be filled in as the project runs._

## Challenges

- _To be filled in as the project runs._
