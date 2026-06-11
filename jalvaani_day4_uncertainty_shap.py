"""
JalVaani AI — Day 4: Conformal Uncertainty Estimation + SHAP Explainability
Centers on the Day 1 stacking ensemble (R2=0.904).

Notes on scope:
- Conformal intervals wrap the FULL stacking ensemble's predictions.
- SHAP uses the XGBoost BASE LEARNER extracted from the ensemble
  (TreeExplainer is exact for trees; the RF + Ridge meta-model are not
  included in the attribution). Documented in output.
- The Day 3 fluoride model is the random-split (Part A) model whose label
  is state-constant; SHAP is expected to expose the state-level lookup.
"""
import gc
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# STEP 1 — RELOAD MODEL AND DATA
# ============================================================
print("STEP 1: LOADING MODEL AND DATA")
FEATURES = joblib.load('jalvaani_features_best.pkl')
print(f"Loaded feature list: {len(FEATURES)} features")

df = pd.read_csv('jalvaani_real_cleaned.csv')
df['lat_lon_interaction'] = df['latitude'] * df['longitude']
df['lat_rounded'] = df['latitude'].round(1)
df['lon_rounded'] = df['longitude'].round(1)
local_median = (df.groupby(['lat_rounded', 'lon_rounded'])['currentlevel']
                  .median().reset_index()
                  .rename(columns={'currentlevel': 'local_median_depth'}))
df = df.merge(local_median, on=['lat_rounded', 'lon_rounded'], how='left')
df['local_median_depth'] = df['local_median_depth'].fillna(df['currentlevel'].median())
station_counts = df.groupby('station_name').size().reset_index(name='station_reading_count')
df = df.merge(station_counts, on='station_name', how='left')
df['station_reading_count'] = df['station_reading_count'].fillna(1)
df['level_diff_abs'] = df['level_diff'].abs()
del local_median, station_counts

season_encoded = pd.get_dummies(df['season'], prefix='season', drop_first=True)
state_encoded = pd.get_dummies(df['state_name'], prefix='state', drop_first=True)
X_full = pd.concat([df[['latitude', 'longitude', 'year_normalized', 'month_sin',
                        'month_cos', 'level_diff_lag', 'day_of_year',
                        'lat_lon_interaction', 'local_median_depth',
                        'station_reading_count', 'level_diff_abs']],
                    season_encoded, state_encoded], axis=1)
X_full = X_full.reindex(columns=FEATURES, fill_value=0.0).astype(np.float32)
valid = X_full.notna().all(axis=1) & df['currentlevel'].notna()
X_full, df = X_full[valid], df[valid]

idx = X_full.sample(n=min(200_000, len(X_full)), random_state=42).index
X_s = X_full.loc[idx]
y_s = df.loc[idx, 'currentlevel'].astype(float)
meta_s = df.loc[idx, ['latitude', 'longitude', 'state_name']]
del X_full

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X_s, y_s, meta_s, test_size=0.2, random_state=42)

print("Loading stacking ensemble (2.5 GB — may take a minute)...")
stacking_model = joblib.load('jalvaani_model_best.pkl')
print("Predicting on test set...")
test_pred = np.asarray(stacking_model.predict(X_test))

# Extract the XGBoost base learner for SHAP, then free the big ensemble
xgb_model = None
for est in stacking_model.estimators_:
    if 'XGB' in type(est).__name__:
        xgb_model = est
        break
assert xgb_model is not None, "XGBRegressor not found in stacking ensemble"
del stacking_model
gc.collect()
print("Models and data loaded. Ready for uncertainty estimation.")

# ============================================================
# STEP 2 — CONFORMAL PREDICTION
# ============================================================
print("\nSTEP 2: CONFORMAL PREDICTION (split-conformal, distribution-free)")
ALPHA = 0.1  # 90% target coverage

# Split test 50/50: calibration / final test
cal_idx, fin_idx = train_test_split(np.arange(len(X_test)),
                                    test_size=0.5, random_state=42)
y_cal, pred_cal = y_test.values[cal_idx], test_pred[cal_idx]
y_fin, pred_fin = y_test.values[fin_idx], test_pred[fin_idx]
meta_fin = meta_test.iloc[fin_idx]
scores = np.abs(y_cal - pred_cal)                 # nonconformity scores
n_cal = len(scores)
print(f"Calibration: {n_cal:,} | Final test: {len(y_fin):,}")


def conformal_q(s, alpha):
    """Finite-sample-corrected conformal quantile."""
    n = len(s)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q, method='higher'))


# --- Marginal intervals
q_hat = conformal_q(scores, ALPHA)
cover_marginal = np.mean(np.abs(y_fin - pred_fin) <= q_hat)
print(f"\nMarginal: q_hat = {q_hat:.3f} mbgl "
      f"(interval = prediction +/- {q_hat:.2f})")
print(f"Marginal coverage on final test: {100 * cover_marginal:.2f}% "
      f"(target >= 90%)")

# --- Adaptive (difficulty-stratified by predicted depth)
BINS = [0, 5, 10, 20, 30, np.inf]
BIN_LABELS = ['0-5', '5-10', '10-20', '20-30', '30+']
cal_bins = pd.cut(pred_cal, bins=BINS, labels=False, include_lowest=True)
fin_bins = pd.cut(pred_fin, bins=BINS, labels=False, include_lowest=True)
# clip predictions below 0 into first bin
cal_bins = np.nan_to_num(cal_bins, nan=0).astype(int)
fin_bins = np.nan_to_num(fin_bins, nan=0).astype(int)

q_hat_adaptive = {}
for b in range(len(BIN_LABELS)):
    s_b = scores[cal_bins == b]
    q_hat_adaptive[b] = conformal_q(s_b, ALPHA) if len(s_b) >= 20 else q_hat

fin_q = np.array([q_hat_adaptive[b] for b in fin_bins])
cover_adaptive = np.mean(np.abs(y_fin - pred_fin) <= fin_q)
print(f"\nAdaptive: per-bin q_hat (90%): "
      + ", ".join(f"{BIN_LABELS[b]}: {q_hat_adaptive[b]:.2f}"
                  for b in range(len(BIN_LABELS))))
print(f"Adaptive coverage on final test: {100 * cover_adaptive:.2f}%")
print("\nCoverage per depth bin (adaptive):")
for b in range(len(BIN_LABELS)):
    m = fin_bins == b
    if m.sum() > 0:
        c = np.mean(np.abs(y_fin[m] - pred_fin[m]) <= fin_q[m])
        print(f"  {BIN_LABELS[b]:>6} mbgl: {100 * c:.1f}% coverage "
              f"({m.sum():,} wells, +/-{q_hat_adaptive[b]:.2f})")

# ============================================================
# STEP 3 — UNCERTAINTY VISUALIZATIONS
# ============================================================
print("\nSTEP 3: UNCERTAINTY VISUALIZATIONS")
order = np.argsort(pred_fin)
rng = np.random.default_rng(42)
pts = rng.choice(len(pred_fin), size=500, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax, q_arr, name in [(axes[0], np.full(len(pred_fin), q_hat), 'Marginal'),
                        (axes[1], fin_q, 'Adaptive (depth-stratified)')]:
    p, q = pred_fin[order], q_arr[order]
    ax.fill_between(np.arange(len(p)), p - q, p + q,
                    color='lightblue', alpha=0.6, label='90% conformal interval')
    ax.plot(np.arange(len(p)), p, color='navy', lw=1, label='Predicted')
    # x-position of each sampled point = its rank in the sorted predictions
    ax.scatter(np.argsort(order)[pts], y_fin[pts], s=6, color='crimson',
               alpha=0.5, label='Actual (sample)')
    ax.set_title(name); ax.set_xlabel('Sample (sorted by predicted depth)')
axes[0].set_ylabel('Groundwater depth (mbgl)')
axes[0].legend(loc='upper left')
fig.suptitle('JalVaani AI — Groundwater Depth Prediction with Conformal '
             'Uncertainty Bounds')
plt.tight_layout(); plt.savefig('uncertainty_intervals.png', dpi=150)
plt.close()

# Interval width distribution by bin
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(BIN_LABELS)))
for b in range(len(BIN_LABELS)):
    m = fin_bins == b
    if m.sum() > 0:
        plt.hist(2 * fin_q[m], bins=30, alpha=0.65, color=colors[b],
                 label=f'{BIN_LABELS[b]} mbgl (width {2 * q_hat_adaptive[b]:.1f})')
plt.xlabel('Interval width (mbgl)'); plt.ylabel('Wells')
plt.title('JalVaani AI — Prediction Uncertainty by Depth Zone')
plt.legend(); plt.tight_layout()
plt.savefig('uncertainty_width_distribution.png', dpi=150)
plt.close()

# Calibration curve
targets = np.linspace(0.5, 0.99, 25)
achieved = []
for t in targets:
    qt = conformal_q(scores, 1 - t)
    achieved.append(np.mean(np.abs(y_fin - pred_fin) <= qt))
plt.figure(figsize=(8, 8))
plt.plot(targets, achieved, 'o-', color='navy', label='Conformal (marginal)')
plt.plot([0.5, 1], [0.5, 1], 'k--', label='Perfect calibration')
plt.xlabel('Target coverage'); plt.ylabel('Achieved coverage')
plt.title('JalVaani AI — Conformal Prediction Calibration Curve')
plt.legend(); plt.tight_layout()
plt.savefig('uncertainty_calibration.png', dpi=150)
plt.close()

# Geographic uncertainty map
midx = rng.choice(len(pred_fin), size=min(15_000, len(pred_fin)), replace=False)
plt.figure(figsize=(11, 11))
sc = plt.scatter(meta_fin['longitude'].values[midx],
                 meta_fin['latitude'].values[midx],
                 c=2 * fin_q[midx], cmap='RdYlGn_r', s=12, alpha=0.6)
plt.colorbar(sc, label='Prediction Uncertainty (mbgl)')
plt.xlabel('Longitude'); plt.ylabel('Latitude')
plt.title('JalVaani AI — Spatial Distribution of Prediction Uncertainty '
          'Across India')
plt.tight_layout(); plt.savefig('uncertainty_geographic_map.png', dpi=150)
plt.close()
print("4 uncertainty plots saved.")

# ============================================================
# STEP 4 — SHAP EXPLAINABILITY (depth model)
# ============================================================
print("\nSTEP 4: SHAP (XGBoost base learner of the stacking ensemble)")
print("NOTE: attributions explain the XGB base learner — exact for trees; "
      "the RF and Ridge meta-model are not included.")

sidx = rng.choice(len(X_test), size=5_000, replace=False)
X_shap = X_test.iloc[sidx].astype(np.float64)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_shap)
xgb_pred_shap = xgb_model.predict(X_shap)
mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
top_depth_feature = mean_abs.idxmax()
print(f"Top SHAP feature (depth): {top_depth_feature} "
      f"(mean |SHAP| = {mean_abs.max():.3f})")

# 4.1 Global importance (bar)
shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=20,
                  show=False)
plt.title('JalVaani AI — Global Feature Importance (SHAP)')
plt.tight_layout(); plt.savefig('shap_global_importance.png', dpi=150)
plt.close()

# 4.2 Beeswarm
shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
plt.title('JalVaani AI — SHAP Beeswarm (direction + magnitude)')
plt.tight_layout(); plt.savefig('shap_beeswarm.png', dpi=150)
plt.close()

# 4.3 Dependence plots, top 4 features
top4 = mean_abs.nlargest(4).index.tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, feat in zip(axes.ravel(), top4):
    shap.dependence_plot(feat, shap_values, X_shap,
                         interaction_index="auto", ax=ax, show=False)
fig.suptitle('JalVaani AI — SHAP Dependence (top 4 features)')
plt.tight_layout(); plt.savefig('shap_dependence_plots.png', dpi=150)
plt.close('all')

# 4.4 Local explanations — 3 contrasting wells (by XGB predicted depth)
def pick(mask):
    cand = np.where(mask)[0]
    return int(cand[0]) if len(cand) else int(np.argmin(xgb_pred_shap))

wells = {
    'Well A — shallow (<5 mbgl)': pick(xgb_pred_shap < 5),
    'Well B — medium (10-20 mbgl)': pick((xgb_pred_shap >= 10) & (xgb_pred_shap <= 20)),
    'Well C — deep (>25 mbgl)': pick(xgb_pred_shap > 25),
}
tmp_files = []
for i, (name, widx) in enumerate(wells.items()):
    expl = shap.Explanation(values=shap_values[widx],
                            base_values=float(explainer.expected_value),
                            data=X_shap.iloc[widx].values,
                            feature_names=FEATURES)
    plt.figure()
    shap.plots.waterfall(expl, max_display=12, show=False)
    plt.title(f"{name}\n(XGB predicted {xgb_pred_shap[widx]:.1f} mbgl)",
              fontsize=10)
    f = f'_tmp_waterfall_{i}.png'
    plt.savefig(f, dpi=150, bbox_inches='tight')
    plt.close('all')
    tmp_files.append(f)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax, f in zip(axes, tmp_files):
    ax.imshow(mpimg.imread(f)); ax.axis('off')
fig.suptitle('JalVaani AI — Individual Well Explanations', fontsize=14)
plt.tight_layout(); plt.savefig('shap_individual_wells.png', dpi=150)
plt.close()
for f in tmp_files:
    os.remove(f)

# 4.5 Interaction: local_median_depth SHAP vs latitude SHAP
i_lmd = FEATURES.index('local_median_depth')
i_lat = FEATURES.index('latitude')
plt.figure(figsize=(9, 7))
sc = plt.scatter(shap_values[:, i_lmd], shap_values[:, i_lat],
                 c=xgb_pred_shap, cmap='RdYlGn_r', s=8, alpha=0.5)
plt.colorbar(sc, label='Predicted depth (mbgl)')
plt.xlabel('SHAP value: local_median_depth')
plt.ylabel('SHAP value: latitude')
plt.title('JalVaani AI — How Location and Baseline Depth Interact')
plt.tight_layout(); plt.savefig('shap_depth_location_interaction.png', dpi=150)
plt.close()
print("5 SHAP plots saved (depth model).")

# ============================================================
# STEP 5 — SHAP FOR THE FLUORIDE MODEL (Day 3)
# ============================================================
print("\nSTEP 5: SHAP — FLUORIDE RISK MODEL")
print("NOTE: this is the Day 3 random-split model (state-constant labels). "
      "SHAP is expected to expose the state-level lookup — an honest "
      "demonstration of WHY the random-split AUC was 1.0.")

fl_model = joblib.load('jalvaani_fluoride_model.pkl')
fl_features = joblib.load('jalvaani_classification_features.pkl')

# Rebuild the Day 3 enriched features (state risk %)
cont = pd.read_csv('RS_Session_267_AU_1211_C_to_D_i.csv')
cont = cont.iloc[:-1].drop(columns=['Sl. No.'])
cont.columns = ['state_name',
                'nitrate_samples', 'nitrate_pct', 'nitrate_districts',
                'fluoride_samples', 'fluoride_pct', 'fluoride_districts',
                'arsenic_samples', 'arsenic_pct', 'arsenic_districts',
                'uranium_samples', 'uranium_pct', 'uranium_districts']
cont = cont.replace('NA', np.nan)
for c in cont.columns[1:]:
    cont[c] = cont[c].astype(float)
cont['state_name'] = cont['state_name'].str.strip().str.title().replace(
    {'Dadra And Nagar Haveli And Daman And Diu':
     'The Dadra And Nagar Haveli And Daman And Diu'})
risk = cont[['state_name', 'fluoride_pct', 'arsenic_pct', 'nitrate_pct',
             'uranium_pct']].rename(columns={
    'fluoride_pct': 'fluoride_risk_pct', 'arsenic_pct': 'arsenic_risk_pct',
    'nitrate_pct': 'nitrate_risk_pct', 'uranium_pct': 'uranium_risk_pct'})

df4 = df.loc[idx].merge(risk, on='state_name', how='left')
for c in ['fluoride_risk_pct', 'arsenic_risk_pct', 'nitrate_risk_pct',
          'uranium_risk_pct']:
    df4[c] = df4[c].fillna(0)
X_fl = pd.concat([X_s.reset_index(drop=True),
                  df4[['fluoride_risk_pct', 'arsenic_risk_pct',
                       'nitrate_risk_pct', 'uranium_risk_pct']]
                  .reset_index(drop=True)], axis=1)
X_fl = X_fl.reindex(columns=fl_features, fill_value=0.0).astype(np.float64)
fidx = rng.choice(len(X_fl), size=3_000, replace=False)
X_fl_s = X_fl.iloc[fidx]

if isinstance(fl_model, Pipeline):           # Logistic Regression pipeline
    sc_ = fl_model.named_steps['scale']
    clf_ = fl_model.named_steps['clf']
    X_fl_t = pd.DataFrame(sc_.transform(X_fl_s), columns=fl_features)
    fl_explainer = shap.LinearExplainer(clf_, X_fl_t)
    fl_shap = fl_explainer.shap_values(X_fl_t)
    plot_X = X_fl_t
else:                                        # tree model
    fl_explainer = shap.TreeExplainer(fl_model)
    fl_shap = fl_explainer.shap_values(X_fl_s)
    if isinstance(fl_shap, list):
        fl_shap = fl_shap[1]
    plot_X = X_fl_s

shap.summary_plot(fl_shap, plot_X, plot_type="bar", max_display=15, show=False)
plt.title('JalVaani AI — What Predicts Fluoride Risk? (SHAP)')
plt.tight_layout(); plt.savefig('shap_fluoride_importance.png', dpi=150)
plt.close()

shap.summary_plot(fl_shap, plot_X, max_display=15, show=False)
plt.title('JalVaani AI — Fluoride Risk SHAP Beeswarm')
plt.tight_layout(); plt.savefig('shap_fluoride_beeswarm.png', dpi=150)
plt.close()

fl_mean_abs = pd.Series(np.abs(fl_shap).mean(axis=0), index=fl_features)
top_fl = fl_mean_abs.nlargest(5)
top_fluoride_feature = top_fl.index[0]
print("\nTop 5 features driving fluoride risk prediction (mean |SHAP|):")
for f, v in top_fl.items():
    print(f"  {f:<45} {v:.4f}")

# ============================================================
# STEP 6 — FULL PROJECT SUMMARY
# ============================================================
print(f"""
==================================================================
            JALVAANI AI — PROJECT SUMMARY
==================================================================
 DAY 1: Groundwater Depth Prediction
   Model: Stacking Ensemble (XGBoost + RF + Ridge)
   Data: 903,658 real CGWB well observations
   R2: 0.9042 | RMSE: 3.775 mbgl | MAE: 1.908 mbgl
   Key feature: {top_depth_feature}
------------------------------------------------------------------
 DAY 2: Physics-Guided Neural Network
   Model: GroundwaterNet + Monotonic Depletion Constraint
   Finding: Corrected monotonic constraint = zero cost
   (+0.024 RMSE, better MAE vs unconstrained NN)
   Finding: Spatial smoothness conflicts with aquifer
   heterogeneity at 0.5 deg resolution
------------------------------------------------------------------
 DAY 3: Contamination Risk Classification
   Data: CGWB Rajya Sabha AU-1211 (official govt data)
   Fluoride AUC: 0.80 (genuine cross-state generalization)
   Arsenic AUC: 0.41 (insufficient positive states)
   Nitrate AUC: 0.50 (agricultural features absent)
   Finding: geology-driven contaminants generalize;
   practice-driven ones require well-level quality data
------------------------------------------------------------------
 DAY 4: Uncertainty + Explainability
   Method: Adaptive Conformal Prediction
   Coverage: {100 * cover_adaptive:.2f}% at 90% target
   SHAP: {top_depth_feature} drives depth predictions
   SHAP: {top_fluoride_feature} drives fluoride risk
==================================================================
""")

# ============================================================
# STEP 7 — SAVE EVERYTHING
# ============================================================
print("STEP 7: SAVE")
joblib.dump(scores, 'jalvaani_conformal_scores.pkl')
joblib.dump(q_hat_adaptive, 'jalvaani_adaptive_qhat.pkl')
np.save('jalvaani_shap_values.npy', shap_values)
pd.DataFrame([{
    'marginal_q_hat': q_hat, 'marginal_coverage': cover_marginal,
    'adaptive_coverage': cover_adaptive,
    'top_depth_feature': top_depth_feature,
    'top_fluoride_feature': top_fluoride_feature,
}]).to_csv('day4_uncertainty_results.csv', index=False)

print(f"""
Day 4 Complete.
Uncertainty: {100 * cover_adaptive:.2f}% coverage at 90% target.
Top SHAP feature (depth): {top_depth_feature}
Top SHAP feature (fluoride): {top_fluoride_feature}
Phase 1 (Days 1-4) complete. Next: well-level water quality integration.""")
