"""
JalVaani AI — Day 5: Multi-Station Groundwater Depletion Forecasting
LSTM vs Transformer, horizons +1/+2/+4 readings (~3/6/12 months).

Methodological notes:
- CGWB readings are approximately (not strictly) quarterly; horizons in
  months are approximate and documented as such.
- Per-station min-max normalization is computed from each station's TRAIN
  portion only (using the full series would leak future range information).
- Splits are by time within each station: the model never trains on a
  sequence whose targets lie in that station's last 20% of readings.
"""
import copy
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print("NOTE: full run takes roughly 1-2 hours on CPU (two sequence models, "
      "up to 50 epochs each; early stopping usually shortens this).\n")

LOOKBACK = 8
HORIZONS = [1, 2, 4]            # steps ahead ~ 3, 6, 12 months
H_NAMES = ['3m', '6m', '12m']
MIN_READINGS = 20
SEASON_ORD = {'monsoon': 0, 'post_monsoon': 1, 'winter': 2, 'pre_monsoon': 3}

# ============================================================
# STEP 1 — DATA PREPARATION FOR TIME SERIES
# ============================================================
print("STEP 1: TIME-SERIES DATA PREPARATION")
df = pd.read_csv('jalvaani_real_cleaned.csv', parse_dates=['date'])
df['lat_rounded'] = df['latitude'].round(1)
df['lon_rounded'] = df['longitude'].round(1)
local_median = (df.groupby(['lat_rounded', 'lon_rounded'])['currentlevel']
                  .median().reset_index()
                  .rename(columns={'currentlevel': 'local_median_depth'}))
df = df.merge(local_median, on=['lat_rounded', 'lon_rounded'], how='left')
df['local_median_depth'] = df['local_median_depth'].fillna(df['currentlevel'].median())
df['level_diff'] = df['level_diff'].fillna(0)
df['level_diff_abs'] = df['level_diff'].abs()
df['season_ord'] = df['season'].map(SEASON_ORD).fillna(0)
del local_median

counts = df['station_name'].value_counts()
keep = counts[counts >= MIN_READINGS].index
df_q = df[df['station_name'].isin(keep)].sort_values(['station_name', 'date'])
n_stations = df_q['station_name'].nunique()
print(f"Stations with >= {MIN_READINGS} readings: {n_stations:,}")

FEAT_COLS = ['level_diff', 'month_sin', 'month_cos', 'year_normalized',
             'season_ord', 'level_diff_abs', 'local_median_depth']

X_list, y_list = [], []
seq_station, seq_i, seq_mn, seq_rng, seq_last_raw, seq_is_test = \
    [], [], [], [], [], []
station_scalers = {}
station_info = {}   # name -> dict(dates, cl, state, lat, lon, split)
last_windows = {}   # name -> (features 8x8, mn, rng, last_raw)

MIN_LEN = LOOKBACK + max(HORIZONS)          # 12
t0 = time.time()
for name, g in df_q.groupby('station_name', sort=False):
    n = len(g)
    if n < MIN_LEN:
        continue
    cl = g['currentlevel'].values.astype(np.float64)
    s = int(0.8 * n)                        # time split point
    mn = float(cl[:s].min())
    rng = float(max(cl[:s].max() - mn, 1e-6))
    cl_norm = (cl - mn) / rng
    feats = g[FEAT_COLS].values.astype(np.float64)
    F = np.column_stack([cl_norm, feats])   # (n, 8)

    station_scalers[name] = (mn, rng)
    station_info[name] = {
        'dates': g['date'].values, 'cl': cl,
        'state': g['state_name'].iloc[0],
        'lat': float(g['latitude'].iloc[0]),
        'lon': float(g['longitude'].iloc[0]), 'split': s}
    last_windows[name] = (F[-LOOKBACK:].copy(), mn, rng, float(cl[-1]))

    for i in range(0, n - MIN_LEN + 1):
        X_list.append(F[i:i + LOOKBACK])
        y_list.append([cl_norm[i + LOOKBACK - 1 + h] for h in HORIZONS])
        seq_station.append(name)
        seq_i.append(i)
        seq_mn.append(mn)
        seq_rng.append(rng)
        seq_last_raw.append(cl[i + LOOKBACK - 1])
        seq_is_test.append(i + LOOKBACK - 1 + max(HORIZONS) >= s)

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)
seq_station = np.asarray(seq_station)
seq_i = np.asarray(seq_i)
seq_mn = np.asarray(seq_mn, dtype=np.float32)
seq_rng = np.asarray(seq_rng, dtype=np.float32)
seq_last_raw = np.asarray(seq_last_raw, dtype=np.float32)
seq_is_test = np.asarray(seq_is_test)
del X_list, y_list
print(f"Sequence building took {time.time() - t0:.0f}s")

tr = ~seq_is_test
te = seq_is_test

# Global scaler for feature columns 1..7, fit on TRAIN sequences only
global_scaler = StandardScaler().fit(X[tr][:, :, 1:].reshape(-1, 7))
X[:, :, 1:] = global_scaler.transform(
    X[:, :, 1:].reshape(-1, 7)).reshape(X.shape[0], LOOKBACK, 7)

print(f"Total sequences created: {len(X):,}")
print(f"Train sequences: {tr.sum():,} | Test sequences: {te.sum():,}")
print(f"Feature shape: (batch, {LOOKBACK} timesteps, {X.shape[2]} features)")
print(f"Target shape: (batch, {y.shape[1]} horizons)")

# 10% of train as validation (random, seed 42)
tr_idx = np.where(tr)[0]
rng_np = np.random.default_rng(42)
rng_np.shuffle(tr_idx)
n_val = int(0.1 * len(tr_idx))
val_idx, train_idx = tr_idx[:n_val], tr_idx[n_val:]
te_idx = np.where(te)[0]

def make_loader(idxs, bs, shuffle):
    ds = TensorDataset(torch.tensor(X[idxs]), torch.tensor(y[idxs]))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

train_loader = make_loader(train_idx, 256, True)
val_loader = make_loader(val_idx, 1024, False)
test_loader = make_loader(te_idx, 1024, False)

# ============================================================
# STEP 2 — LSTM MODEL
# ============================================================
class GroundwaterLSTM(nn.Module):
    def __init__(self, n_features=8, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            dropout=0.2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, len(HORIZONS)))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])     # last hidden state


# ============================================================
# STEP 3 — TRANSFORMER MODEL
# ============================================================
class GroundwaterTransformer(nn.Module):
    def __init__(self, n_features=8, d_model=64, seq_len=LOOKBACK):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, len(HORIZONS)))

    def forward(self, x):
        z = self.encoder(self.proj(x) + self.pos)
        return self.head(z.mean(dim=1))     # global average pooling


for cls in (GroundwaterLSTM, GroundwaterTransformer):
    m = cls()
    print(f"{cls.__name__} trainable parameters: "
          f"{sum(p.numel() for p in m.parameters() if p.requires_grad):,}")

# ============================================================
# STEP 4 — TRAINING
# ============================================================
def train_model(model, name, epochs=50, patience=10):
    print(f"\n===== Training {name} =====")
    model = model.to(DEVICE)
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {'train': [], 'val': [], 'lr': []}
    best_val, best_state, bad = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(train_idx)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        vl /= len(val_idx)
        sched.step()
        lr = opt.param_groups[0]['lr']
        hist['train'].append(tl); hist['val'].append(vl); hist['lr'].append(lr)
        print(f"Epoch {ep:3d}/{epochs} | train {tl:.5f} | val {vl:.5f} "
              f"| lr {lr:.2e}")
        if vl < best_val - 1e-6:
            best_val, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep}")
                break
    model.load_state_dict(best_state)
    return model, hist


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.vstack(out)


lstm_model, lstm_hist = train_model(GroundwaterLSTM(), 'LSTM')
tf_model, tf_hist = train_model(GroundwaterTransformer(), 'Transformer')

plt.figure(figsize=(10, 6))
plt.plot(lstm_hist['train'], color='#1f77b4', label='LSTM train')
plt.plot(lstm_hist['val'], color='#1f77b4', ls='--', label='LSTM val')
plt.plot(tf_hist['train'], color='#d62728', label='Transformer train')
plt.plot(tf_hist['val'], color='#d62728', ls='--', label='Transformer val')
plt.xlabel('Epoch'); plt.ylabel('MSE loss (normalized)')
plt.title('JalVaani AI — Forecasting Model Training Curves')
plt.legend(); plt.tight_layout()
plt.savefig('forecasting_loss_curves.png', dpi=150)
plt.close()

# ============================================================
# STEP 5 — EVALUATION
# ============================================================
print("\nSTEP 5: EVALUATION (real mbgl units)")
def inverse(pred_norm, idxs):
    return pred_norm * seq_rng[idxs, None] + seq_mn[idxs, None]

y_test_mbgl = inverse(y[te_idx], te_idx)
lstm_pred = inverse(predict(lstm_model, test_loader), te_idx)
tf_pred = inverse(predict(tf_model, test_loader), te_idx)
naive_pred = np.repeat(seq_last_raw[te_idx, None], len(HORIZONS), axis=1)

rows = []
for h, hname in enumerate(H_NAMES):
    a = y_test_mbgl[:, h]
    for mname, p in [('Naive', naive_pred), ('LSTM', lstm_pred),
                     ('Transformer', tf_pred)]:
        rows.append({
            'Horizon': hname, 'Model': mname,
            'RMSE': float(np.sqrt(mean_squared_error(a, p[:, h]))),
            'MAE': float(mean_absolute_error(a, p[:, h])),
            'R2': float(r2_score(a, p[:, h]))})
res = pd.DataFrame(rows)
res.to_csv('day5_forecasting_results.csv', index=False)

print("\n--- MODEL COMPARISON ---")
print(f"{'Horizon':<9}{'LSTM RMSE':>11}{'Transf RMSE':>13}"
      f"{'LSTM R2':>10}{'Transf R2':>11}")
for hname in H_NAMES:
    r = res[res.Horizon == hname].set_index('Model')
    print(f"{hname:<9}{r.loc['LSTM', 'RMSE']:>11.4f}"
          f"{r.loc['Transformer', 'RMSE']:>13.4f}"
          f"{r.loc['LSTM', 'R2']:>10.4f}{r.loc['Transformer', 'R2']:>11.4f}")
print("\n--- VS NAIVE BASELINE (predict last observed) ---")
print(f"{'Horizon':<9}{'Naive RMSE':>12}{'LSTM RMSE':>11}{'Transf RMSE':>13}")
for hname in H_NAMES:
    r = res[res.Horizon == hname].set_index('Model')
    print(f"{hname:<9}{r.loc['Naive', 'RMSE']:>12.4f}"
          f"{r.loc['LSTM', 'RMSE']:>11.4f}"
          f"{r.loc['Transformer', 'RMSE']:>13.4f}")

rmse12 = res[(res.Horizon == '12m')].set_index('Model')['RMSE']
best_name = 'LSTM' if rmse12['LSTM'] <= rmse12['Transformer'] else 'Transformer'
best_model = lstm_model if best_name == 'LSTM' else tf_model
best_pred = lstm_pred if best_name == 'LSTM' else tf_pred
print(f"\nBest model by 12-month RMSE: {best_name}")

# ============================================================
# STEP 7 (computed early for plots) — CONFORMAL ON FORECASTS
# ============================================================
print("\nSTEP 7: CONFORMAL PREDICTION INTERVALS (calibrated on validation)")
val_loader_seq = make_loader(val_idx, 1024, False)
val_pred_n = predict(best_model, val_loader_seq)
val_pred = inverse(val_pred_n, val_idx)
val_act = inverse(y[val_idx], val_idx)

def conformal_q(s, alpha=0.1):
    n = len(s)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q, method='higher'))

q_hats = {hname: conformal_q(np.abs(val_act[:, h] - val_pred[:, h]))
          for h, hname in enumerate(H_NAMES)}
for h, hname in enumerate(H_NAMES):
    cov = np.mean(np.abs(y_test_mbgl[:, h] - best_pred[:, h]) <= q_hats[hname])
    print(f"  {hname}: q_hat = {q_hats[hname]:.2f} mbgl | "
          f"test coverage {100 * cov:.1f}% (target 90%)")

# ============================================================
# STEP 6 — FORECASTING VISUALIZATIONS
# ============================================================
print("\nSTEP 6: VISUALIZATIONS")

# Pick 6 representative stations (most test sequences per state group)
test_stations = pd.Series(seq_station[te_idx]).value_counts()
def pick_station(states, taken):
    for st in test_stations.index:
        if st not in taken and station_info[st]['state'] in states:
            return st
    return None

picked, taken = [], set()
for states in [['Punjab'], ['Rajasthan'], ['Madhya Pradesh', 'Gujarat'],
               ['Maharashtra', 'Telangana'], ['Kerala'], ['West Bengal']]:
    st = pick_station(states, taken)
    if st:
        picked.append(st); taken.add(st)
print(f"Stations plotted: {[(s, station_info[s]['state']) for s in picked]}")


def station_forecast_plot(fname, with_uncertainty):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, st in zip(axes.ravel(), picked):
        info = station_info[st]
        dates, cl, s = info['dates'], info['cl'], info['split']
        ax.plot(dates, cl, color='gray', lw=1, alpha=0.7, label='Observed')
        ax.axvline(dates[s], color='black', ls=':', lw=1.5,
                   label='Train/test split')
        m = (seq_station == st) & seq_is_test
        sidx = np.where(m)[0]
        if len(sidx):
            t_pos = seq_i[sidx] + LOOKBACK       # index of +1-step target
            t_pos = np.clip(t_pos, 0, len(dates) - 1)
            d = dates[t_pos]
            pos_in_te = np.searchsorted(te_idx, sidx)
            act = y_test_mbgl[pos_in_te, 0]
            lp = lstm_pred[pos_in_te, 0]
            tp = tf_pred[pos_in_te, 0]
            bp = best_pred[pos_in_te, 0]
            ax.plot(d, act, 'o-', color='black', ms=3, lw=1, label='Actual (test)')
            ax.plot(d, lp, 's--', color='#1f77b4', ms=3, lw=1, label='LSTM 3m')
            ax.plot(d, tp, '^--', color='#d62728', ms=3, lw=1, label='Transformer 3m')
            if with_uncertainty:
                ax.fill_between(d, bp - q_hats['3m'], bp + q_hats['3m'],
                                color='orange', alpha=0.25,
                                label=f'{best_name} 90% interval')
        ax.set_title(f"{st[:35]} ({info['state']})", fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    axes[0, 0].legend(fontsize=7, loc='best')
    fig.suptitle('JalVaani AI — Groundwater Depletion Forecasts by Station '
                 '(LSTM vs Transformer)' +
                 (' with Conformal Uncertainty' if with_uncertainty else ''))
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


station_forecast_plot('forecasting_station_plots.png', with_uncertainty=False)
station_forecast_plot('forecasting_with_uncertainty.png', with_uncertainty=True)

# Horizon degradation
plt.figure(figsize=(8, 6))
for mname, color in [('Naive', 'gray'), ('LSTM', '#1f77b4'),
                     ('Transformer', '#d62728')]:
    vals = [res[(res.Horizon == h) & (res.Model == mname)]['RMSE'].iloc[0]
            for h in H_NAMES]
    plt.plot(H_NAMES, vals, 'o-', color=color, label=mname)
plt.xlabel('Forecast horizon'); plt.ylabel('RMSE (mbgl)')
plt.title('JalVaani AI — Forecast Error Growth with Horizon')
plt.legend(); plt.tight_layout()
plt.savefig('forecasting_horizon_degradation.png', dpi=150)
plt.close()

# Depletion trend map (true future forecast from each station's last window)
print("Computing future forecasts for all stations...")
names = list(last_windows.keys())
F_last = np.stack([last_windows[n][0] for n in names]).astype(np.float32)
F_last[:, :, 1:] = global_scaler.transform(
    F_last[:, :, 1:].reshape(-1, 7)).reshape(len(names), LOOKBACK, 7)
mns = np.array([last_windows[n][1] for n in names], dtype=np.float32)
rngs = np.array([last_windows[n][2] for n in names], dtype=np.float32)
lasts = np.array([last_windows[n][3] for n in names], dtype=np.float32)
fut = []
with torch.no_grad():
    for i in range(0, len(F_last), 4096):
        fut.append(best_model(torch.tensor(F_last[i:i + 4096]).to(DEVICE))
                   .cpu().numpy())
fut = np.vstack(fut) * rngs[:, None] + mns[:, None]
trend = fut[:, 2] - lasts                       # 12m forecast - last observed

lats = np.array([station_info[n]['lat'] for n in names])
lons = np.array([station_info[n]['lon'] for n in names])
states = np.array([station_info[n]['state'] for n in names])

vmax = np.percentile(np.abs(trend), 95)
plt.figure(figsize=(11, 11))
sc = plt.scatter(lons, lats, c=trend, cmap='RdBu_r',
                 norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                 s=10, alpha=0.6)
plt.colorbar(sc, label='Predicted 12-month depth change (mbgl)\n'
                       'red = depleting, blue = recovering')
plt.xlabel('Longitude'); plt.ylabel('Latitude')
plt.title('JalVaani AI — Predicted 12-Month Groundwater Trend Across India')
plt.tight_layout(); plt.savefig('forecasting_depletion_map.png', dpi=150)
plt.close()

# Error by state (best model, 12m horizon, per-station RMSE)
err_df = pd.DataFrame({
    'station': seq_station[te_idx],
    'err2': (y_test_mbgl[:, 2] - best_pred[:, 2]) ** 2})
st_rmse = np.sqrt(err_df.groupby('station')['err2'].mean())
st_state = pd.Series({n: station_info[n]['state'] for n in st_rmse.index})
err_state = pd.DataFrame({'rmse': st_rmse, 'state': st_state})
top_states = err_state['state'].value_counts().nlargest(12).index
plt.figure(figsize=(14, 6))
data = [err_state.loc[err_state.state == s, 'rmse'].values for s in top_states]
plt.boxplot(data, labels=top_states, showfliers=False)
plt.xticks(rotation=30, ha='right')
plt.ylabel(f'Per-station 12m RMSE (mbgl, {best_name})')
plt.title('JalVaani AI — Forecast Difficulty by State')
plt.tight_layout(); plt.savefig('forecasting_error_by_state.png', dpi=150)
plt.close()
print("All forecasting plots saved.")

# ============================================================
# STEP 8 — SAVE
# ============================================================
print("\nSTEP 8: SAVE")
torch.save(lstm_model.state_dict(), 'jalvaani_lstm_forecaster.pth')
torch.save(tf_model.state_dict(), 'jalvaani_transformer_forecaster.pth')
joblib.dump(station_scalers, 'jalvaani_station_scalers.pkl')
joblib.dump(global_scaler, 'jalvaani_global_scaler_day5.pkl')

forecasts_df = pd.DataFrame({
    'station_name': names, 'state': states, 'lat': lats, 'lon': lons,
    'last_observed': lasts,
    'forecast_3m': fut[:, 0], 'forecast_6m': fut[:, 1],
    'forecast_12m': fut[:, 2],
    'trend_direction': np.where(trend > 0.5, 'depleting',
                        np.where(trend < -0.5, 'recovering', 'stable'))})
forecasts_df.to_csv('jalvaani_forecasts.csv', index=False)

n_depleting = int((forecasts_df['trend_direction'] == 'depleting').sum())
state_trend = pd.DataFrame({'state': states, 'trend': trend}) \
    .groupby('state')['trend'].mean()
state_trend = state_trend[pd.Series(states).value_counts() >= 20]
worst_state = state_trend.idxmax()

print(f"""
Day 5 Complete. JalVaani Depletion Forecasting done.
Stations forecasted: {len(names):,}
Best model: {best_name}
3-month RMSE: {res[(res.Horizon=='3m') & (res.Model==best_name)]['RMSE'].iloc[0]:.4f} mbgl
6-month RMSE: {res[(res.Horizon=='6m') & (res.Model==best_name)]['RMSE'].iloc[0]:.4f} mbgl
12-month RMSE: {res[(res.Horizon=='12m') & (res.Model==best_name)]['RMSE'].iloc[0]:.4f} mbgl
Stations showing depletion trend: {n_depleting:,} / {len(names):,}
Most at-risk state (highest avg predicted depletion, >=20 stations): {worst_state}""")
