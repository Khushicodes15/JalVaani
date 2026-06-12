"""
JalVaani Day 5 — LSTM depletion forecasting.

Primary path: lookup from jalvaani_forecasts.csv (16,693 pre-computed station forecasts).
Conformal intervals from Day 5 global q_hat (calibrated on validation sequences;
test coverage 82-84% vs 90% target — temporal distribution-shift caveat applies).
"""
import math
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

_forecasts_df: Optional[pd.DataFrame] = None
_station_scalers: Optional[dict] = None   # {station_name: (min, max)}

DATA = Path(__file__).parent.parent / "data"
SAVED = Path(__file__).parent.parent / "saved_models"

QHAT = {"3m": 2.43, "6m": 2.77, "12m": 3.05}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(a, 1.0)))


def load() -> bool:
    global _forecasts_df, _station_scalers
    path = DATA / "jalvaani_forecasts.csv"
    if not path.exists():
        return False
    _forecasts_df = pd.read_csv(path)
    _forecasts_df["_name_lower"] = _forecasts_df["station_name"].str.lower().str.strip()
    # Load station min-max scalers for OOD quality check
    scalers_path = SAVED / "jalvaani_station_scalers.pkl"
    if scalers_path.exists():
        with open(scalers_path, "rb") as f:
            _station_scalers = pickle.load(f)
    return True


def _quality_check(station_name: str, last_observed: float,
                   fc3: float, fc6: float, fc12: float) -> tuple:
    """
    Returns (quality, note, spread).

    quality = "low_confidence" when:
      - last_observed is outside the station's training scaler range (OOD input), OR
      - horizon spread > 5 mbgl (forecasts are inconsistent across horizons).

    38.6% of stations in the dataset have last_observed outside training range,
    typically because water levels shifted significantly in the held-out test period.
    These forecasts should be treated as indicative only.
    """
    spread = round(max(fc3, fc6, fc12) - min(fc3, fc6, fc12), 3)
    notes = []
    quality = "reliable"

    if _station_scalers:
        sc = _station_scalers.get(station_name)
        if sc is not None:
            smin, smax = min(float(sc[0]), float(sc[1])), max(float(sc[0]), float(sc[1]))
            if last_observed < smin or last_observed > smax:
                quality = "low_confidence"
                norm_input = (last_observed - smin) / (smax - smin) if smax != smin else 0
                notes.append(
                    f"Last observation ({last_observed:.2f} mbgl) is outside this "
                    f"station's training range [{smin:.2f}–{smax:.2f} mbgl] "
                    f"(normalized input = {norm_input:.2f}; model is extrapolating)."
                )

    if spread > 5.0:
        quality = "low_confidence"
        notes.append(
            f"Horizon spread is {spread:.1f} mbgl — the 3/6/12-month forecasts "
            f"are inconsistent, suggesting unstable model behaviour at this station."
        )

    if not notes:
        note = "Last observation within training range and horizons are consistent."
    else:
        note = " ".join(notes)

    return quality, note, spread


def _interval(value: float, q: float) -> dict:
    return {
        "value": round(float(value), 3),
        "lower": round(max(0.0, float(value) - q), 3),
        "upper": round(float(value) + q, 3),
    }


def forecast(station_name: str) -> Optional[dict]:
    """Look up pre-computed forecast. Returns None if station not found."""
    if _forecasts_df is None:
        raise RuntimeError("Forecast data not loaded")
    mask = _forecasts_df["_name_lower"] == station_name.lower().strip()
    if not mask.any():
        return None
    row = _forecasts_df[mask].iloc[0]
    sname = str(row["station_name"])
    last_obs = round(float(row["last_observed"]), 3)
    fc3 = float(row["forecast_3m"])
    fc6 = float(row["forecast_6m"])
    fc12 = float(row["forecast_12m"])
    quality, note, spread = _quality_check(sname, last_obs, fc3, fc6, fc12)
    return {
        "station_name": sname,
        "last_observed_depth": last_obs,
        "forecast_3_month": _interval(fc3, QHAT["3m"]),
        "forecast_6_month": _interval(fc6, QHAT["6m"]),
        "forecast_12_month": _interval(fc12, QHAT["12m"]),
        "trend": str(row["trend_direction"]),
        "forecast_quality": quality,
        "forecast_quality_note": note,
        "horizon_spread_mbgl": spread,
    }


def nearest_forecast_stations(lat: float, lon: float, n: int = 5) -> list:
    """Return n nearest stations from the forecast dataset."""
    if _forecasts_df is None:
        return []
    dists = _forecasts_df.apply(
        lambda r: _haversine(lat, lon, float(r["lat"]), float(r["lon"])), axis=1
    )
    idx = dists.nsmallest(n).index
    return [
        {
            "station_name": str(_forecasts_df.loc[i, "station_name"]),
            "state": str(_forecasts_df.loc[i, "state"]),
            "distance_km": round(float(dists[i]), 1),
        }
        for i in idx
    ]


def get_national_stats() -> dict:
    """Aggregate depletion statistics for /stats/national."""
    if _forecasts_df is None:
        return {}
    total = len(_forecasts_df)
    depleting = int((_forecasts_df["trend_direction"] == "depleting").sum())
    by_state = (
        _forecasts_df.groupby("state")["trend_direction"]
        .apply(lambda x: round(float((x == "depleting").mean()), 3))
        .sort_values(ascending=False)
    )
    return {
        "forecasted_stations": total,
        "stations_depleting": depleting,
        "depletion_pct": round(depleting / total * 100, 1),
        "top_5_depleting_states": by_state.head(5).to_dict(),
    }
