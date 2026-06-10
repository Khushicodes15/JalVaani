import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')

print("LOADING DATA...")
df = pd.read_csv('jalvaani_real_cleaned.csv')

def get_base_X(data):
    season_encoded = pd.get_dummies(data['season'], prefix='season', drop_first=True)
    state_encoded = pd.get_dummies(data['state_name'], prefix='state', drop_first=True)
    cols = ['latitude', 'longitude', 'year_normalized', 'month_sin', 'month_cos', 'level_diff_lag', 'day_of_year']
    X = pd.concat([data[cols], season_encoded, state_encoded], axis=1)
    return X

y = df['currentlevel']
X_base = get_base_X(df)

# Sample 200,000 rows
sample_idx = X_base.sample(n=min(200000, len(X_base)), random_state=42).index
X_sampled = X_base.loc[sample_idx]
y_sampled = y.loc[sample_idx]

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

results = [
    {'Model': 'Original Linear Regression', 'RMSE': 10.1542, 'MAE': 6.6168, 'R2': 0.3070},
    {'Model': 'Original Random Forest', 'RMSE': 4.4958, 'MAE': 2.3202, 'R2': 0.8641},
    {'Model': 'Original XGBoost', 'RMSE': 6.4299, 'MAE': 3.8525, 'R2': 0.7221}
]

print("\n--- IMPROVEMENT 1 — Better XGBoost tuning ---")
X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

tuned_xgb = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

tuned_xgb.fit(X_tr_xgb, y_tr_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)
y_pred_tuned_xgb = tuned_xgb.predict(X_test)
rmse_tuned_xgb = np.sqrt(mean_squared_error(y_test, y_pred_tuned_xgb))
mae_tuned_xgb = mean_absolute_error(y_test, y_pred_tuned_xgb)
r2_tuned_xgb = r2_score(y_test, y_pred_tuned_xgb)

print(f"Tuned XGBoost -> RMSE: {rmse_tuned_xgb:.4f}, MAE: {mae_tuned_xgb:.4f}, R2: {r2_tuned_xgb:.4f}")
results.append({'Model': 'Tuned XGBoost', 'RMSE': rmse_tuned_xgb, 'MAE': mae_tuned_xgb, 'R2': r2_tuned_xgb})


print("\n--- IMPROVEMENT 2 — Add these missing features to X ---")
# Feature 1
df['lat_lon_interaction'] = df['latitude'] * df['longitude']

# Feature 2
df['lat_rounded'] = df['latitude'].round(1)
df['lon_rounded'] = df['longitude'].round(1)
local_median = df.groupby(['lat_rounded', 'lon_rounded'])['currentlevel'].median().reset_index()
local_median.rename(columns={'currentlevel': 'local_median_depth'}, inplace=True)
df = df.merge(local_median, on=['lat_rounded', 'lon_rounded'], how='left')

df['local_median_depth'] = df['local_median_depth'].fillna(df['currentlevel'].median())

# Feature 3
station_counts = df.groupby('station_name').size().reset_index(name='station_reading_count')
df = df.merge(station_counts, on='station_name', how='left')
df['station_reading_count'] = df['station_reading_count'].fillna(1)

# Feature 4
if 'level_diff' in df.columns:
    df['level_diff_abs'] = df['level_diff'].abs()
else:
    df['level_diff_abs'] = df['level_diff_lag'].abs()

# Rebuild X
X_new_base = get_base_X(df)
new_features = df[['lat_lon_interaction', 'local_median_depth', 'station_reading_count', 'level_diff_abs']]
X_new = pd.concat([X_new_base, new_features], axis=1)

X_sampled_new = X_new.loc[sample_idx]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_sampled_new, y_sampled, test_size=0.2, random_state=42)

# Free full-size frames (909k rows) — only the 200k sample is needed from here on
X_base_cols = list(X_base.columns)
X_new_cols = list(X_new.columns)
import gc
del X_new, X_new_base, new_features, X_base, df, local_median, station_counts
gc.collect()

# Retrain RF
rf_new = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_new.fit(X_train_new, y_train_new)
y_pred_rf_new = rf_new.predict(X_test_new)
rmse_rf_new = np.sqrt(mean_squared_error(y_test_new, y_pred_rf_new))
mae_rf_new = mean_absolute_error(y_test_new, y_pred_rf_new)
r2_rf_new = r2_score(y_test_new, y_pred_rf_new)
print(f"Random Forest + new features -> RMSE: {rmse_rf_new:.4f}, MAE: {mae_rf_new:.4f}, R2: {r2_rf_new:.4f}")
results.append({'Model': 'Random Forest + new features', 'RMSE': rmse_rf_new, 'MAE': mae_rf_new, 'R2': r2_rf_new})

# Retrain Tuned XGBoost
X_tr_xgb_new, X_val_xgb_new, y_tr_xgb_new, y_val_xgb_new = train_test_split(X_train_new, y_train_new, test_size=0.1, random_state=42)
tuned_xgb_new = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=8, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, early_stopping_rounds=20
)
tuned_xgb_new.fit(X_tr_xgb_new, y_tr_xgb_new, eval_set=[(X_val_xgb_new, y_val_xgb_new)], verbose=False)
y_pred_xgb_new = tuned_xgb_new.predict(X_test_new)
rmse_xgb_new = np.sqrt(mean_squared_error(y_test_new, y_pred_xgb_new))
mae_xgb_new = mean_absolute_error(y_test_new, y_pred_xgb_new)
r2_xgb_new = r2_score(y_test_new, y_pred_xgb_new)
print(f"Tuned XGBoost + new features -> RMSE: {rmse_xgb_new:.4f}, MAE: {mae_xgb_new:.4f}, R2: {r2_xgb_new:.4f}")
results.append({'Model': 'Tuned XGBoost + new features', 'RMSE': rmse_xgb_new, 'MAE': mae_xgb_new, 'R2': r2_xgb_new})


print("\n--- IMPROVEMENT 3 — Stacking ensemble ---")
estimators = [
    ('xgb', xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=8, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
]

# n_jobs=1: fit estimators sequentially in the main process. n_jobs=-1 spawns
# worker processes that pickle the fitted RF + data copies -> MemoryError.
# Base models still parallelize internally via their own n_jobs=-1.
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=1
)

print("Fitting Stacking Ensemble on full train set (internal 5-fold CV)...")
try:
    stacking_model.fit(X_train_new, y_train_new)
    y_pred_stack = stacking_model.predict(X_test_new)

    rmse_stack = np.sqrt(mean_squared_error(y_test_new, y_pred_stack))
    mae_stack = mean_absolute_error(y_test_new, y_pred_stack)
    r2_stack = r2_score(y_test_new, y_pred_stack)

    print(f"Stacking Ensemble -> RMSE: {rmse_stack:.4f}, MAE: {mae_stack:.4f}, R2: {r2_stack:.4f}")
    results.append({'Model': 'Stacking Ensemble', 'RMSE': rmse_stack, 'MAE': mae_stack, 'R2': r2_stack})
except MemoryError:
    print("WARNING: Stacking Ensemble ran out of memory — skipping it. Best of the remaining models will be saved.")

print("\n--- COMPARISON TABLE ---")
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv('improvement_results.csv', index=False)

# Determine best model
best_row = res_df.loc[res_df['RMSE'].idxmin()]
best_model_name = best_row['Model']
best_rmse = best_row['RMSE']
best_r2 = best_row['R2']

if best_model_name == 'Stacking Ensemble':
    best_model_obj = stacking_model
elif best_model_name == 'Random Forest + new features':
    best_model_obj = rf_new
elif best_model_name == 'Tuned XGBoost + new features':
    best_model_obj = tuned_xgb_new
elif best_model_name == 'Tuned XGBoost':
    best_model_obj = tuned_xgb
else:
    best_model_obj = rf_new

joblib.dump(best_model_obj, 'jalvaani_model_best.pkl')
if 'new features' in best_model_name or 'Stacking' in best_model_name:
    joblib.dump(X_new_cols, 'jalvaani_features_best.pkl')
else:
    joblib.dump(X_base_cols, 'jalvaani_features_best.pkl')

print(f"\nBest model: {best_model_name}, R²: {best_r2:.4f}, RMSE: {best_rmse:.4f} mbgl")
