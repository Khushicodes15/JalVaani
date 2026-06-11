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

`jalvaani_day5_forecasting.py` forecasts station-level groundwater depth at ~3, 6 and 12 months ahead (readings are approximately quarterly) using two sequence models trained under identical settings: a 2-layer LSTM and a 3-layer Transformer encoder. Stations with ≥20 readings contribute sliding 8-reading windows; splits are time-based within each station (last 20% = test) and per-station normalization is fit on train portions only, so no future information leaks into training. A naive last-value baseline sets the minimum bar, and conformal prediction (calibrated on validation sequences) provides 90% forecast intervals. Outputs include `jalvaani_forecasts.csv` — true future forecasts per station with 12-month trend direction — and a national depletion trend map. *(Results pending current training run.)*

## Setup

```bash
pip install -r requirements.txt
pip install torch          # Days 2 and 5
pip install shap matplotlib-venn  # Day 4 / Day 3 extras
```

Raw CGWB / State Groundwater Board CSVs and large trained binaries are not tracked in git (see `.gitignore`). Place `cgwb_water_level.csv` and `state_water_level.csv` in the repo root, then run:

```bash
python run_real_pipeline.py      # Day 1: clean, EDA, baselines
python improve_models.py        # Day 1: features + stacking ensemble
python jalvaani_day2_physics.py  # Day 2: physics-guided NN + ablation
python jalvaani_day2_corrected.py # Day 2: corrected constraints
python jalvaani_day2_finalize.py  # Day 2: train + save official model
python jalvaani_day3_contamination.py    # Day 3: contamination risk + LOSO eval
python jalvaani_day4_uncertainty_shap.py # Day 4: conformal uncertainty + SHAP
python jalvaani_day5_forecasting.py      # Day 5: LSTM/Transformer forecasting
```

## Artifacts

`jalvaani_physicsnn_day2_final.pth` — official Day 2 model weights (GroundwaterNet, Version B). Load with the scalers produced by the Day 2 pipeline. Evaluation plots (`*.png`) and ablation metrics (`*results*.csv`) are tracked in the repo.
