English | [Português](README.pt-BR.md)

# Cortex Analytics Platform

![status](https://img.shields.io/badge/status-completed%20(technical%20challenge%2C%202026--06)-lightgrey)
![backend](https://img.shields.io/badge/backend-FastAPI-009688)
![frontend](https://img.shields.io/badge/frontend-React-61DAFB)
![ml](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![license](https://img.shields.io/badge/license-MIT-green)

A full-stack analytics platform built as a technical challenge for a developer position.
It ingests industrial asset telemetry (temperature, vibration, pressure, runtime hours),
runs a Pandas ETL and two scikit-learn models (anomaly detection and failure risk), and
shows everything in a React dashboard.

## Status

**Completed (technical challenge, 2026-06).** Not continued after delivery. In 2026-11 the
failure risk model got a holdout evaluation, an evaluation script and tests.

## The problem

Machine telemetry is often scattered, with no central monitoring and no predictive layer.
Failures are only noticed once the equipment has stopped. The challenge asked for a single
layer that ingests telemetry, computes operational metrics, raises alerts and estimates
failure risk per asset.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Front end | React + Vite + TailwindCSS + Recharts | fast build, time series charts out of the box |
| Back end | FastAPI (Python) | same language as the data pipeline; OpenAPI at `/docs` |
| Database | PostgreSQL (docker-compose) or SQLite (local default) | SQLite runs without Docker |
| ETL | Pandas | per-asset aggregation in one pass |
| ML | scikit-learn: Isolation Forest + Random Forest | anomalies without labels; risk with feature importance |
| Infra | Docker + docker-compose | database, API and front end with one command |

```
raw events ──► ETL (Pandas) ──► database ──► ML models ──► dashboard
 (sensors)     aggregate, load   metrics      anomaly, risk  KPIs and charts
                                 and alerts
```

## Failure risk model evaluation

**The data is synthetic.** `backend/seed.py` generates 5 assets with 60 readings each
(seed 42). Three assets are healthy and two degrade over time. A reading is labeled `down`
by a fixed rule: temperature above 92 or vibration above 9. That gives 300 readings, 23 of
them failures (7.7%), all in two assets.

The first version trained and scored on the same rows. Now the model is evaluated on a
stratified 75/25 holdout with a fixed seed, and only then retrained on all data to score
the assets. The response of `POST /api/ml/failures/run` includes the evaluation.

Output of `python avaliar.py`:

| Evaluation | test n (failures) | ROC-AUC | Precision | Recall | Confusion (TN, FP, FN, TP) |
|---|---|---|---|---|---|
| Old method: train = test (do not use) | 300 (23) | 1.000 | 1.000 | 1.000 | 277, 0, 0, 23 |
| **Stratified 75/25 holdout** | **75 (6)** | **1.000** | **1.000** | **0.833** | **69, 0, 1, 5** |
| The seed's own rule, same test set | 75 (6) | 1.000 | 1.000 | 1.000 | 69, 0, 0, 6 |
| Without temperature and vibration | 75 (6) | 0.949 | 0.333 | 0.167 | 67, 2, 5, 1 |
| Two whole assets held out for test | 120 (12) | 0.985 | 1.000 | 0.500 | 108, 0, 6, 6 |

How to read it:

- **The high number says nothing about the real world.** The label is a rule over two of
  the features, and the rule alone scores 100%. What the holdout shows is that the Random
  Forest rediscovers the rule on readings it has not seen (it missed 1 of 6 failures).
- **The test set has only 6 failures.** One hit more or less moves recall by 17 points.
  The confidence interval is wide and the number should not be quoted without this context.
- **Without the two rule features, recall drops to 0.167.** AUC stays high (0.949) because
  the degrading assets also have higher pressure in the generator: the model separates the
  assets, not the moment of failure.
- **With whole assets held out, recall drops to 0.500.** A new asset is harder than new
  readings from a known asset.
- **The label is the current state, not a future failure.** To become a real prediction,
  the label would have to look ahead (for example: "was there a `down` in the next 24 h?"),
  and the data would have to be real.

Feature importance in the final model: temperature 0.573, vibration 0.238, pressure 0.153,
runtime hours 0.036. This also mirrors the generator's rule.

## Running locally

### Without Docker (SQLite)

```bash
git clone https://github.com/pethrus07/desafio-tecnico-dev-.git
cd desafio-tecnico-dev-/backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python seed.py                  # synthetic demo data (drops and recreates the database)
uvicorn app.main:app --reload   # API at http://localhost:8000, docs at /docs

# in another terminal
cd ../frontend
npm install
npm run dev                     # dashboard at http://localhost:5173
```

### With Docker (PostgreSQL)

```bash
docker compose up --build
docker compose exec backend python seed.py
# frontend: http://localhost:5173 | backend: http://localhost:8000
```

In the dashboard, the **"Rodar ETL + Modelos"** button (Portuguese UI: "Run ETL +
models") runs the ETL and both models and fills KPIs, telemetry, predictions and alerts.

## Tests and evaluation

```bash
cd backend
python -m pytest     # 9 tests: seed, idempotent ETL, holdout and both models
python avaliar.py    # the evaluation report in the table above
```

Tests use in-memory SQLite and never touch the local database or Postgres. The front end
has no tests.

## Layout

```
backend/
  app/
    main.py              FastAPI, CORS and routers
    database.py          engine (Postgres or SQLite) and session
    models.py            Asset, Event, Metric, Alert, Prediction
    routers/             events, metrics/etl, alerts, predictions (ML)
    etl/pipeline.py      idempotent Pandas ETL
    ml/anomaly.py        Isolation Forest
    ml/failure_prediction.py   Random Forest with holdout
  seed.py                synthetic data generator
  avaliar.py             failure risk model evaluation
  tests/                 pytest
frontend/
  src/App.jsx            dashboard
  src/components/        KPICard, MetricsChart, AlertsPanel, PredictionsPanel
docker-compose.yml
```

Code comments and the UI are in Portuguese.

## Architecture

Use cases, process and C4 in [docs/arquitetura/](docs/arquitetura/README.md) (Portuguese).

## Known limits

- Synthetic data; the metric measures the method, not performance on a real plant.
- No migrations: the schema comes from `create_all`, and the seed drops and recreates it.
- The model is retrained on every request, nothing is persisted.
- The ETL reads the whole events table; there is no time window.
- The telemetry chart uses list position on the X axis and mixes assets.
- CORS is open for development.
- Never deployed to production.

## Roadmap

Not continued. Real time, observability and ingestion at scale were in the original plan
and were not built.

## License

MIT, see [LICENSE](LICENSE).
