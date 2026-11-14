# Architecture — Drift-Aware ML API

```mermaid
flowchart LR
    Client --> API[FastAPI ml-api/app.py]
    API --> Model[Iris / Churn model]
    API --> Log[prediction log (parquet)]
    Log --> Drift[Evidently drift monitor]
    Drift --> Report[GET /drift-report]
    Drift --> Retrain[GitHub Actions retrain workflow]
    Retrain --> Review[human review]
    Review --> Model
```

## Components

- `ml-api/app.py` — FastAPI: `GET /health`, `POST /predict`,
  `GET /drift-report`. Prediction logging to a local append-only
  parquet file (Iris/Churn features only, no personal data).
- `ml-api/train.py` (Day 2) — train Iris (then Churn), save model +
  `reference_data.parquet` (the training distribution Evidently
  compares against).
- `ml-api/drift_monitor.py` (Day 5) — Evidently AI: compare recent
  logged inputs vs reference; drift score + per-feature drift table.
- `.github/workflows/retrain.yml` (Day 6) — `workflow_dispatch`
  retrain: re-run training, evaluate, open a PR with the new model
  only if accuracy on a held-out set does not drop.

## Decisions

- **FastAPI, not Flask**, for this service: async + automatic
  request validation fit a prediction API better.
- **Drift triggers a proposal, not a swap.** A retrained model is
  reviewed before it replaces the old one — a silent swap would hide
  a bad retrain.
- **Free tier only.** Local + Docker first; Hugging Face Spaces or
  Render free tier for a demo link. No paid hosting.
