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

## Setup

```bash
pip install -r requirements.txt
pip install torch          # Day 2 only
```

Raw CGWB / State Groundwater Board CSVs and large trained binaries are not tracked in git (see `.gitignore`). Place `cgwb_water_level.csv` and `state_water_level.csv` in the repo root, then run:

```bash
python run_real_pipeline.py      # Day 1: clean, EDA, baselines
python improve_models.py        # Day 1: features + stacking ensemble
python jalvaani_day2_physics.py  # Day 2: physics-guided NN + ablation
python jalvaani_day2_corrected.py # Day 2: corrected constraints
python jalvaani_day2_finalize.py  # Day 2: train + save official model
```

## Artifacts

`jalvaani_physicsnn_day2_final.pth` — official Day 2 model weights (GroundwaterNet, Version B). Load with the scalers produced by the Day 2 pipeline. Evaluation plots (`*.png`) and ablation metrics (`*results*.csv`) are tracked in the repo.
