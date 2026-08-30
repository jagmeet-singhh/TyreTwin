#  TyreTwin — Formula 1 Tyre Degradation & Noise Isolation Intelligence Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TyreTwin** is an AI-powered Formula 1 pit wall telemetry intelligence platform designed to isolate **true tyre degradation** from noisy practice session telemetry (FP1, FP2, FP3) by decoupling external influences (fuel load burn, traffic wake, track evolution, weather, driver aggression) and forecasting degradation curves with probabilistic $\pm 2\sigma$ Gaussian Process confidence intervals.

---

## The Motorsport Problem

In Formula 1 practice sessions, observed lap times do **not** directly reflect tyre wear due to compounding external noise factors:

$$	ext{Observed Lap Time} = 	ext{True Tyre Performance} + 	ext{External Noise}$$

Where **External Noise** consists of:
- **Fuel Mass Burn**: Starting ~45-105kg ($10	ext{kg} pprox 0.30	ext{s}$ lap penalty).
- **Track Grip Evolution**: Rubbering-in curve ($rac{t}{T_{	ext{total}}}$) recovering up to $0.45	ext{s}$ over a session.
- **Traffic & Dirty Air**: Wake turbulence and micro-sector speed drops adding $+0.65	ext{s}+$.
- **Driver Push Aggression**: Throttle & brake severity delta.
- **Thermal Conditions**: Track temperature offset from optimal tyre operating window.

---

## Two-Stage Machine Learning Architecture

```mermaid
graph TD
    A[FastF1 Practice Telemetry FP1/FP2/FP3] --> B[Data Preprocessor & IQR Cleaning]
    B --> C[Motorsport Feature Engineering]
    C --> D[Model 1: Gradient Boosting Noise Isolation]
    D -->|Clean Lap Times: Observed - Predicted Noise| E[Model 2: Gaussian Process Regressor]
    E -->|Probabilistic Curves ±2σ| F[Strategy Assistant & Pit Window Solver]
    E --> G[SHAP Explainable AI Waterfall]
    F --> H[FastAPI REST Backend]
    G --> H
    H --> I[Streamlit Pit Wall Command Center]
    H --> J[ReportLab PDF Debrief Exporter]
```

### Stage 1: Noise Isolation Model (Gradient Boosting)
Predicts external non-tyre noise from physics-engineered features (fuel burn, track evolution index, traffic flag, push aggression index, thermal delta):

$$\widehat{	ext{Noise}} = f_{	ext{GBR}}(	ext{Fuel}, 	ext{Evolution}, 	ext{Push}, 	ext{Traffic}, 	ext{Thermal})$$
$$	ext{Clean Lap Time} = 	ext{Observed Lap Time} - \widehat{	ext{Noise}}$$

### Stage 2: Probabilistic Degradation Model (Gaussian Process Regression)
Models tyre wear progression using a Matérn kernel with uncertainty bounds:

$$k(x, x') = \sigma_0^2 	ext{Matérn}_{5/2}(r) + \sigma_{	ext{noise}}^2$$
$$\widehat{y}(	ext{lap}) \sim \mathcal{N}\left(\mu(	ext{lap}), \sigma^2(	ext{lap})ight)$$

Yields smooth degradation curves with shaded $\pm 2\sigma$ confidence bands (95% uncertainty envelope).

---

##  Repository Structure

```text
TyreTwin/
├── app/                                 # Streamlit Pit-Wall Frontend
│   ├── dashboard.py                     # Main Command Center Overview
│   ├── pages/
│   │   ├── 1__Degradation_Intelligence.py  # Gaussian Process Curves & SHAP
│   │   ├── 2__Telemetry_Explorer.py        # Synchronized 100Hz Multi-Sensor Trace
│   │   ├── 3__Strategy_Assistant.py         # Pit Window Solver & Crossover Deltas
│   │   ├── 4__Race_Validation.py            # Post-Race Model Accuracy Benchmarking
│   │   ├── 5__Engineering_Report.py         # 1-Click Executive PDF Export
│   │   └── 6__Live_Stint_Replay.py          # Turn-by-Turn Telemetry Playback
│   ├── components/                      # MetricCard, ConfidenceGauge, Selectors, StrategyCard
│   ├── charts/                          # Plotly interactive visualizers
│   └── utils/                           # F1 Dark Theme CSS, REST API Client, PDF Generator
│
├── backend/                             # FastAPI Backend Microservice
│   ├── main.py                          # Application entrypoint & CORS middleware
│   ├── config.py                        # Pydantic settings & environment configuration
│   ├── api/endpoints/                   # /sessions, /predict, /strategy, /validate, /telemetry, /explain, /report
│   ├── database/                        # Normalized SQLAlchemy models & SQLite/Postgres auto-fallback
│   ├── schemas/                         # Pydantic v2 schemas
│   └── services/                        # Strategy, Data, and ML orchestrators
│
├── ml/                                  # Motorsport ML Pipeline
│   ├── fastf1_loader.py                 # FastF1 ingestion & cached benchmark telemetry
│   ├── preprocessing.py                 # In/Out lap filtering & IQR outlier removal
│   ├── feature_engineering.py           # Fuel burn, track evolution, traffic detection, push aggression
│   ├── train_noise.py                   # Model 1: Gradient Boosting Noise Isolation
│   ├── train_degradation.py             # Model 2: Gaussian Process Regressor with Matérn Kernel
│   ├── predict.py                       # Unified prediction inference engine
│   ├── validation.py                    # Evaluation metrics (MAE, RMSE, R²)
│   └── explainability.py                # SHAP-style waterfall attribution
│
├── docker/                              # Multi-stage Dockerfiles (Backend & Frontend)
├── tests/                               # 100% passing Pytest suite
├── docker-compose.yml                   # Containerized stack orchestration
├── requirements.txt                     # Pinned dependencies
├── run.py                               # Single-command concurrent launcher
└── README.md                            # Complete engineering documentation
```

---

##  Quickstart Guide

### Option 1: Single-Command Native Launch (Recommended)

1. Clone or navigate to the repository:
   ```bash
   cd TyreTwin
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch both FastAPI Backend and Streamlit Dashboard concurrently:
   ```bash
   python run.py
   ```

4. Access the platforms:
   - **Streamlit Pit Wall Dashboard**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2: Docker Compose Deployment

Launch the complete microservice architecture (PostgreSQL database, FastAPI backend, Streamlit frontend):

```bash
docker compose up --build
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/sessions/load` | Ingests and synchronizes FastF1 session telemetry with DB caching. |
| `GET` | `/api/sessions/list` | Returns list of available cached race sessions. |
| `GET` | `/api/sessions/drivers` | Returns metadata for 2024 drivers and teams. |
| `POST` | `/api/predict` | Computes decoupled clean lap times, GP degradation curves, and $\pm 2\sigma$ bounds. |
| `POST` | `/api/strategy` | Recommends optimal pit stop laps, stint lengths, and crossover points. |
| `POST` | `/api/validate` | Benchmarks model predictions against actual stint pace ($MAE, RMSE, R^2$). |
| `GET` | `/api/telemetry` | Retrieves 100Hz synchronized traces (Speed, Throttle, Brake, RPM, Gear, DRS). |
| `POST` | `/api/explain` | Generates SHAP-style feature attribution for lap time noise factors. |
| `POST` | `/api/report/generate` | Compiles official Pit Wall PDF debrief sheet using ReportLab. |
| `GET` | `/api/health` | Health check probe. |

---

##  Automated Testing

Execute the test suite with pytest:

```bash
pytest tests/ -v
```

---

##  Tech Stack

- **Frontend**: Streamlit, Plotly, Custom F1 Pit-Wall Dark CSS.
- **Backend**: FastAPI, Uvicorn, Pydantic v2, SQLAlchemy.
- **ML & Telemetry**: FastF1, XGBoost, Scikit-learn (GaussianProcessRegressor), SHAP, Pandas, NumPy.
- **Reporting**: ReportLab PDF Engine.
- **Deployment**: Docker & Docker Compose.
