"""
JalVaani AI API — shared utilities.

Feature engineering mirrors training exactly:
  - year_normalized = (year - 2013) / 10.0  (training span: 2013–2021)
  - season dummies: post_monsoon / pre_monsoon / winter  (monsoon = base/dropped)
  - month->season: Jan-Mar=winter, Apr-May=pre_monsoon, Jun-Sep=monsoon, Oct-Dec=post_monsoon
  - state dummies: 31 states, prefix "state_"
  - 45 features from jalvaani_features_best.pkl
  - 49 features for contamination (adds 4 risk_pct columns)
"""
import math
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

YEAR_MIN = 2013
YEAR_SCALE = 10.0

SEASON_MAP = {
    1: "winter", 2: "winter", 3: "winter",
    4: "pre_monsoon", 5: "pre_monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon", 12: "post_monsoon",
}

# Adaptive conformal depth bins: [0-5, 5-10, 10-20, 20-30, 30+] → keys 0-4
DEPTH_BIN_EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, float("inf")]

# Day-5 global conformal q_hat per horizon
DAY5_QHAT = {"3m": 2.43, "6m": 2.77, "12m": 3.05}


# ── Haversine ─────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(a, 1.0)))


# ── Station lookup ────────────────────────────────────────────────────────────

def find_nearest_station(lat: float, lon: float, station_df: pd.DataFrame) -> dict:
    """Find nearest row in station_df by haversine distance. Returns a dict."""
    dists = station_df.apply(
        lambda r: haversine(lat, lon, float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )
    idx = dists.idxmin()
    row = station_df.loc[idx].to_dict()
    row["distance_km"] = round(float(dists[idx]), 2)
    return row


# ── Uncertainty ───────────────────────────────────────────────────────────────

def get_depth_bin(depth: float) -> int:
    """Return adaptive conformal bin index 0–4."""
    for i in range(len(DEPTH_BIN_EDGES) - 1):
        if DEPTH_BIN_EDGES[i] <= depth < DEPTH_BIN_EDGES[i + 1]:
            return i
    return len(DEPTH_BIN_EDGES) - 2


def get_uncertainty_interval(prediction: float, adaptive_qhat: dict) -> dict:
    """
    Return 90% conformal prediction interval and uncertainty label.
    adaptive_qhat: {0: 2.36, 1: 4.31, 2: 6.46, 3: 7.89, 4: 10.84}
    """
    bin_idx = get_depth_bin(prediction)
    q = float(adaptive_qhat.get(bin_idx, adaptive_qhat.get(4, 5.0)))
    lower = round(max(0.0, prediction - q), 3)
    upper = round(prediction + q, 3)
    width = q * 2
    if width < 3:
        level = "low"
    elif width < 6:
        level = "medium"
    else:
        level = "high"
    return {"lower": lower, "upper": upper, "q_hat": round(q, 3), "level": level}


# ── Feature engineering ───────────────────────────────────────────────────────

def build_feature_vector(
    lat: float,
    lon: float,
    date_str: str,
    station_data: dict,
    feature_names: List[str],
    risk_lookup: Dict[str, dict],
) -> np.ndarray:
    """
    Build a feature vector in the exact column order of feature_names.

    Works for both:
      - 45-feature depth model  (jalvaani_features_best.pkl)
      - 49-feature contamination model  (jalvaani_classification_features.pkl)

    Missing features (e.g. out-of-training states) default to 0.0.

    Caveat: level_diff_lag uses the station's historical median from the lookup.
    For year outside 2013–2021 training range, year_normalized is clamped to [0, 0.8].
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    month = dt.month
    doy = float(dt.timetuple().tm_yday)

    # Clamp year_normalized to training range to avoid wild extrapolation
    year_norm = float(np.clip((year - YEAR_MIN) / YEAR_SCALE, 0.0, 0.8))
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    season = SEASON_MAP[month]

    lat_lon_interaction = lat * lon
    local_median_depth = float(station_data.get("local_median_depth") or 10.0)
    station_reading_count = float(station_data.get("station_reading_count") or 50.0)
    level_diff_lag = float(station_data.get("level_diff_lag_median") or 0.0)
    level_diff_abs = abs(level_diff_lag)

    state = str(station_data.get("state_name", ""))
    risks = risk_lookup.get(
        state,
        {"fluoride_risk_pct": 0.0, "arsenic_risk_pct": 0.0,
         "nitrate_risk_pct": 0.0, "uranium_risk_pct": 0.0},
    )

    scalar_map = {
        "latitude": lat,
        "longitude": lon,
        "year_normalized": year_norm,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "level_diff_lag": level_diff_lag,
        "day_of_year": doy,
        "lat_lon_interaction": lat_lon_interaction,
        "local_median_depth": local_median_depth,
        "station_reading_count": station_reading_count,
        "level_diff_abs": level_diff_abs,
        "fluoride_risk_pct": float(risks.get("fluoride_risk_pct", 0.0)),
        "arsenic_risk_pct": float(risks.get("arsenic_risk_pct", 0.0)),
        "nitrate_risk_pct": float(risks.get("nitrate_risk_pct", 0.0)),
        "uranium_risk_pct": float(risks.get("uranium_risk_pct", 0.0)),
    }

    row = []
    for fname in feature_names:
        if fname in scalar_map:
            row.append(scalar_map[fname])
        elif fname.startswith("season_"):
            row.append(1.0 if season == fname[len("season_"):] else 0.0)
        elif fname.startswith("state_"):
            row.append(1.0 if state == fname[len("state_"):] else 0.0)
        else:
            row.append(0.0)

    return np.array(row, dtype=np.float64)
