# JalVaani AI

AI-powered groundwater intelligence platform for India. Predicts water-table depth (meters below ground level, mbgl) from ~909,000 real monitoring readings collected by the Central Ground Water Board (CGWB) and State Groundwater Boards.

## Results

| Model | RMSE (mbgl) | MAE (mbgl) | R² |
|---|---|---|---|
| **Day 1: Stacking Ensemble (XGBoost + RF + Ridge)** | **3.775** | **1.908** | **0.904** |
| Day 2: Physics-Guided NN (monotonic constraint) | 5.009 | 2.622 | 0.831 |
| Baseline: Random Forest | 4.496 | 2.320 | 0.864 |
| Baseline: Linear Regression | 10.154 | 6.617 | 0.307 |

## Day 1 — Tabular ML Pipeline

`run_real_pipeline.py` merges and cleans both data sources, engineers temporal features (cyclical month encoding, seasons) and trains baseline models. `improve_models.py` adds spatial-temporal features — local median depth via lat/lon grid clustering, station reliability counts, lat-lon interaction — and a stacking ensemble (tuned XGBoost + Random Forest, Ridge meta-model, 5-fold CV), lifting R² from 0.864 to 0.904.

## Day 2 — Physics-Guided Neural Network

`jalvaani_model_architecture.py` defines **GroundwaterNet** (deep MLP with skip connection) and **CorrectedPhysicsLoss**, which augments MSE with two soft physical constraints: monotonic depletion (depth increases over time within a 0.1° grid cell) and spatial smoothness (nearby locations, similar depths).

**Research finding (ablation study):** the monotonic depletion constraint, applied within hydrologically coherent grid cells, achieves physical consistency at near-zero accuracy cost (+0.024 RMSE, improved MAE). The spatial smoothness prior conflicts with real aquifer heterogeneity at 0.5° resolution and degrades accuracy; it is excluded from the final model. Naive batch-level constraints (applied across unrelated stations) degrade performance substantially — physics constraints in groundwater ML must respect hydrological unit boundaries. Full analysis in `jalvaani_day2_final.ipynb`.

## Day 3 — Contamination Risk Classification

`jalvaani_day3_contamination.py` integrates real CGWB state-level contamination statistics (Rajya Sabha Session 267, AU-1211) and classifies fluoride, arsenic and nitrate risk. Two evaluations are reported: a random-split pipeline (AUC 1.0 by construction, since labels are state-constant — documented as a data-integration check) and the headline **leave-states-out** evaluation, where entire states are held out and state-identifying features excluded.

**Research finding:** fluoride risk generalizes to unseen states (AUC 0.80) because it is geologically driven; arsenic (AUC 0.41, only 3 positive states) and nitrate (AUC 0.50, agriculture-driven) do not — state-aggregate labels support cross-state generalization only for geology-determined contaminants. Full analysis in `jalvaani_day3_final.ipynb`.

## Day 4 — Conformal Uncertainty + SHAP Explainability

`jalvaani_day4_uncertainty_shap.py` wraps the Day 1 ensemble with **split-conformal prediction** (distribution-free, finite-sample valid): 90.5% empirical coverage at a 90% target, with depth-adaptive intervals from ±2.4 mbgl (shallow wells) to ±10.8 mbgl (deep wells). SHAP analysis of the XGBoost base learner identifies `local_median_depth` as the dominant predictor; SHAP applied to the Day 3 fluoride classifier correctly exposes its state-level lookup structure (`fluoride_risk_pct` dominates), corroborating the Day 3 evaluation design.

## Day 5 — Multi-Station Depletion Forecasting

`jalvaani_day5_forecasting.py` forecasts station-level groundwater depth at ~3, 6 and 12 months ahead (readings are approximately quarterly) using two sequence models trained under identical settings: a 2-layer LSTM and a 3-layer Transformer encoder. Stations with ≥20 readings contribute sliding 8-reading windows; splits are time-based within each station (last 20% = test) and per-station normalization is fit on train portions only, so no future information leaks into training. A naive last-value baseline sets the minimum bar, and conformal prediction (calibrated on validation sequences) provides forecast intervals. Both models beat the naive baseline on all three horizons (≥40% RMSE reduction). LSTM is the stronger model at longer ranges (12m RMSE **3.889 mbgl**, R² **0.879**); the Transformer is marginally better at 3m (RMSE 3.171). Of 16,693 forecasted stations, **5,868 (35%) show a predicted depletion trend**; Telangana has the highest average predicted depletion among states with ≥20 stations. Conformal test coverage was 82–84% against a 90% target — a documented limitation of split-conformal under temporal distribution shift.

## Day 6 — FastAPI Backend

`jalvaani_api/` wraps all five days into a single REST API (FastAPI + uvicorn). Models are loaded once at startup; endpoints are stateless. The `/report/full` endpoint integrates depth prediction, contamination risk, depletion forecast, and a physics consistency check into one response with a human-readable summary.

Nine endpoints: `POST /predict/depth`, `POST /predict/contamination`, `GET /forecast/{station_name}`, `POST /report/full`, `GET /stations`, `GET /stations/search?q=`, `GET /stats/national`, `GET /health`, `GET /`. Interactive documentation auto-generated at `http://localhost:8000/docs`.

Backend hardening: every endpoint is `async def` with CPU-bound inference offloaded via `asyncio.to_thread()`; sliding-window rate limiting (120 RPM per IP); GZip compression; thread-safe TTL cache (1 hr national stats, 5 min station pages); env-based config via pydantic-settings.

## Day 7 — Production React UI

`jalvaani_ui/` is a React 18 + Vite 5 + TypeScript + Tailwind CSS frontend connecting to the Day 6 API. Six pages: Dashboard (national stats + depletion chart), Depth Predictor, Contamination Risk (SVG gauge dials), Station Forecast (LSTM timeline with shaded CI band), Full Report, and Station Explorer. Indian water/earth colour palette; responsive to all screen sizes; code-split by route.

Infrastructure: `gunicorn.conf.py` (UvicornWorker + `preload_app=True` for Linux COW memory sharing), `Dockerfile` (multi-stage Node → Python), `docker-compose.yml`, `nginx.conf` (static asset caching, upstream keepalive, request rate limiting).

## Setup

**Research pipeline (Days 1–5):**

```bash
pip install -r requirements.txt
pip install torch shap matplotlib-venn
# Place cgwb_water_level.csv and state_water_level.csv in the repo root, then:
python run_real_pipeline.py
python improve_models.py
python jalvaani_day2_physics.py && python jalvaani_day2_corrected.py && python jalvaani_day2_finalize.py
python jalvaani_day3_contamination.py
python jalvaani_day4_uncertainty_shap.py
python jalvaani_day5_forecasting.py
```

**API (Day 6):**

```bash
# Copy all .pkl/.pth to jalvaani_api/saved_models/ and CSVs to jalvaani_api/data/
pip install -r jalvaani_api/requirements.txt
cd jalvaani_api && uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs
```

**Full stack with UI (Day 7):**

```bash
# Terminal 1 — API
cd jalvaani_api && uvicorn main:app --reload --port 8000

# Terminal 2 — UI dev server (proxies API calls to :8000)
cd jalvaani_ui && npm install && npm run dev
# → http://localhost:5173

# Or build for production:
cd jalvaani_ui && npm run build
# Set STATIC_DIR=../jalvaani_ui/dist in .env, then:
cd .. && gunicorn -c gunicorn.conf.py jalvaani_api.main:app
# → http://localhost:8000
```

**Docker (single command):**

```bash
docker compose up --build
# → http://localhost:8000
```

## Artifacts

`jalvaani_physicsnn_day2_final.pth` — official Day 2 model weights (GroundwaterNet, Version B). `jalvaani_conformal_scores.pkl` + `jalvaani_adaptive_qhat.pkl` — Day 4 conformal artifacts (reproduce uncertainty intervals without retraining). Evaluation plots (`*.png`) and results files (`*results*.csv`) are tracked in the repo. Large binaries (2.5 GB stacking ensemble, LSTM/Transformer weights, station scalers) are not tracked — regenerate by running the pipeline scripts.
