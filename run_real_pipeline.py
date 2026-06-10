import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("STEP 1: MERGE AND CLEAN")
df1 = pd.read_csv('cgwb_water_level.csv')
df1['source_board'] = 'CGWB'

df2 = pd.read_csv('state_water_level.csv')
df2['source_board'] = 'State'

df = pd.concat([df1, df2], ignore_index=True)
print(f"Initial shape after concat: {df.shape}")

# Parse date
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear

# Filter currentlevel
df = df[df['currentlevel'].notnull() & (df['currentlevel'] > 0) & (df['currentlevel'] <= 100)]

# Filter state_name
df = df[df['state_name'].notnull()]

df.reset_index(drop=True, inplace=True)
print(f"Final shape after cleaning: {df.shape}")
print(df.head())

print("\nSTEP 2: FEATURE ENGINEERING")
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['year_normalized'] = (df['year'] - 2013) / (2023 - 2013)

def get_season(m):
    if m in [6, 7, 8, 9]: return 'monsoon'
    elif m in [10, 11, 12]: return 'post_monsoon'
    elif m in [1, 2, 3]: return 'winter'
    elif m in [4, 5]: return 'pre_monsoon'
    return 'unknown'

df['season'] = df['month'].apply(get_season)

# One-hot encoding
season_encoded = pd.get_dummies(df['season'], prefix='season', drop_first=True)
state_encoded = pd.get_dummies(df['state_name'], prefix='state', drop_first=True)

df['level_diff_lag'] = df['level_diff']

# Combine features
cols = ['latitude', 'longitude', 'year_normalized', 'month_sin', 'month_cos', 'level_diff_lag', 'day_of_year']
X = pd.concat([df[cols], season_encoded, state_encoded], axis=1)
y = df['currentlevel']

print(f"X.shape: {X.shape}")
print(f"X.columns: {X.columns.tolist()}")

print("\nSTEP 3: EDA")
# 1. Line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='year', y='currentlevel', hue='season', estimator=np.median, errorbar=None)
plt.title('JalVaani AI — Groundwater Depth Trend by Season (Real CGWB Data)')
plt.ylabel('Median Current Level (mbgl)')
plt.tight_layout()
plt.savefig('groundwater_depth_trend_season.png')
plt.close()

# 2. Heatmap
top_states = df['state_name'].value_counts().nlargest(15).index
heatmap_data = df[df['state_name'].isin(top_states)].pivot_table(
    values='currentlevel', index='state_name', columns='season', aggfunc=np.median
)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('JalVaani AI — State vs Season Groundwater Depth')
plt.tight_layout()
plt.savefig('state_season_heatmap.png')
plt.close()

# 3. Scatter plot
plt.figure(figsize=(8, 6))
sample_df = df.sample(n=min(20000, len(df)), random_state=42)
sns.scatterplot(data=sample_df, x='latitude', y='currentlevel', hue='season', alpha=0.1)
plt.title('JalVaani AI — Geographic Distribution of Water Depth')
plt.ylabel('Current Level (mbgl)')
plt.tight_layout()
plt.savefig('geographic_water_depth.png')
plt.close()

# 4. Distribution plot
plt.figure(figsize=(8, 6))
sns.histplot(df['currentlevel'], kde=True, bins=50, color='teal')
plt.title('JalVaani AI — Distribution of Water Table Depth (Real Data)')
plt.xlabel('Current Level (mbgl)')
plt.tight_layout()
plt.savefig('distribution_water_depth.png')
plt.close()

print("\nSTEP 4: TRAIN MODELS")
# Subsample for training
train_sample_idx = X.sample(n=min(200000, len(X)), random_state=42).index
X_sampled = X.loc[train_sample_idx]
y_sampled = y.loc[train_sample_idx]

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
}

results = []
best_model = None
best_rmse = float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name
        best_r2 = r2
        
    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}\n")

results_df = pd.DataFrame(results)

# Comparison Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Model', y='RMSE', palette='viridis')
plt.title('JalVaani AI — Model Comparison (RMSE)')
plt.ylabel('RMSE (mbgl)')
plt.tight_layout()
plt.savefig('real_model_comparison.png')
plt.close()

print("\nSTEP 5: PREDICTION VISUALIZATION")
# Best Model predictions for plotting
X_test_plot = X_test.copy()
y_test_plot = y_test.copy()
y_pred_best = best_model.predict(X_test_plot)

# 1. Actual vs Predicted
plt.figure(figsize=(8, 6))
plot_idx = np.random.choice(len(y_test_plot), size=min(5000, len(y_test_plot)), replace=False)
plt.scatter(y_test_plot.iloc[plot_idx], y_pred_best[plot_idx], alpha=0.3, color='blue')
min_val = min(y_test_plot.min(), y_pred_best.min())
max_val = max(y_test_plot.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title('JalVaani AI — Actual vs Predicted (Best Model)')
plt.xlabel('Actual Current Level (mbgl)')
plt.ylabel('Predicted Current Level (mbgl)')
plt.tight_layout()
plt.savefig('real_actual_vs_predicted.png')
plt.close()

# 2. Residuals
residuals = y_test_plot - y_pred_best
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_best, residuals, alpha=0.3, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('JalVaani AI — Residuals vs Predicted (Best Model)')
plt.xlabel('Predicted Current Level (mbgl)')
plt.ylabel('Residuals')
plt.tight_layout()
plt.savefig('real_residuals.png')
plt.close()

# 3. Geographic prediction map
plt.figure(figsize=(10, 8))
map_idx = np.random.choice(len(X_test_plot), size=min(10000, len(X_test_plot)), replace=False)
sc = plt.scatter(X_test_plot.iloc[map_idx]['longitude'], X_test_plot.iloc[map_idx]['latitude'], 
                 c=y_pred_best[map_idx], cmap='RdYlGn_r', alpha=0.6, s=15)
plt.colorbar(sc, label='Predicted Depth (mbgl)')
plt.title('JalVaani AI — Predicted Groundwater Depth Across India')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('india_groundwater_map.png')
plt.close()

print("\nSTEP 6: FEATURE IMPORTANCE")
if best_model_name == 'XGBoost':
    importances = best_model.feature_importances_
    feature_names = X.columns
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 8))
    colors = ['red'] if len(feat_imp_df) > 0 else []
    if len(feat_imp_df) > 1:
        colors += ['steelblue'] * (len(feat_imp_df) - 1)
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette=colors)
    plt.title('JalVaani AI — Drivers of Groundwater Depth (Real CGWB Data)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('real_feature_importance.png')
    plt.close()

print("\nSTEP 7: SAVE")
joblib.dump(best_model, 'jalvaani_model_day1_real.pkl')
joblib.dump(list(X.columns), 'jalvaani_features_day1_real.pkl')
df.to_csv('jalvaani_real_cleaned.csv', index=False)

print(f"\nDay 1 Complete — Real Data. Records used: {len(df)}. Best model: {best_model_name}. R²: {best_r2:.4f}. RMSE: {best_rmse:.4f} mbgl.")
