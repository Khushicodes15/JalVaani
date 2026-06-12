"""
JalVaani AI API — FastAPI backend.

Production-ready features:
  - Async endpoints: CPU-bound ML inference offloaded via asyncio.to_thread
    so the event loop stays free for I/O (supports thousands of concurrent
    connections per process; scale horizontally with Gunicorn + UvicornWorker
    + preload_app=True for memory-efficient model sharing via Linux COW fork).
  - GZip compression for large JSON responses (>500 bytes).
  - In-process sliding-window rate limiting per IP (120 RPM default).
  - In-process TTL caching for /stats/national and /stations pages.
  - Env-based CORS (see config.py / .env).
  - Optional SPA static file serving: set STATIC_DIR=../jalvaani_ui/dist.

Run (development):
  uvicorn main:app --reload --port 8000

Run (production, single machine):
  gunicorn -c ../gunicorn.conf.py main:app

Docs: http://localhost:8000/docs
"""
import asyncio
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Ensure jalvaani_api/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from cache import cache
from config import settings
from schemas import (
    ContaminationResponse,
    DepthPredictionResponse,
    ForecastResponse,
    FullReportResponse,
    LocationInput,
)
import utils
import models.depth_model as depth_model
import models.physics_model as physics_model
import models.contamination_model as contamination_model
import models.forecasting_model as forecasting_model

BASE  = Path(__file__).parent
SAVED = BASE / "saved_models"
DATA  = BASE / "data"


# ── Shared application state (populated once at startup) ──────────────────────

_state: dict = {
    "station_df":    None,   # pd.DataFrame — one row per station
    "adaptive_qhat": None,   # dict {0..4: float}
    "risk_lookup":   None,   # dict {state_name: {fluoride_risk_pct, ...}}
    "models_loaded": [],     # list[str] of successfully loaded model names
}


# ── Startup helpers ───────────────────────────────────────────────────────────

def _load_station_lookup() -> None:
    csv_path = DATA / "jalvaani_real_cleaned.csv"
    if not csv_path.exists():
        print("[startup] WARNING: jalvaani_real_cleaned.csv not found in data/")
        return
    raw = pd.read_csv(
        csv_path,
        usecols=["station_name", "state_name", "district_name",
                 "latitude", "longitude", "currentlevel", "level_diff_lag"],
    )
    raw["lat_cell"] = (raw["latitude"] * 10).round() / 10
    raw["lon_cell"] = (raw["longitude"] * 10).round() / 10
    cell_med = (
        raw.groupby(["lat_cell", "lon_cell"])["currentlevel"]
        .median()
        .rename("local_median_depth")
    )
    raw = raw.join(cell_med, on=["lat_cell", "lon_cell"])
    rc = raw["station_name"].value_counts().rename("station_reading_count")
    raw = raw.join(rc, on="station_name")
    _state["station_df"] = (
        raw.groupby("station_name")
        .agg(
            state_name=("state_name", "first"),
            district_name=("district_name", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            local_median_depth=("local_median_depth", "first"),
            station_reading_count=("station_reading_count", "first"),
            level_diff_lag_median=("level_diff_lag", "median"),
        )
        .reset_index()
    )
    print(f"[startup] Station lookup: {len(_state['station_df'])} stations")


def _load_risk_lookup() -> None:
    rs_path = DATA / "RS_Session_267_AU_1211_C_to_D_i.csv"
    if not rs_path.exists():
        print("[startup] WARNING: RS_Session CSV not found — contamination features default to 0")
        _state["risk_lookup"] = {}
        return
    cdf = pd.read_csv(rs_path)
    cdf = cdf[cdf["State/UT"] != "Grand Total"].copy()
    cdf["state_name"] = cdf["State/UT"].str.title().replace(
        "Dadra And Nagar Haveli And Daman And Diu",
        "The Dadra And Nagar Haveli And Daman And Diu",
    )
    cdf = cdf.replace("NA", np.nan)
    col_map = {
        "Fluoride - Percent of samples with F >1.5 mg/L":  "fluoride_risk_pct",
        "Arsenic - Percent of samples with As> 10 ppb":    "arsenic_risk_pct",
        "Nitrate - Percent of samples with NO3> 45 mg/L":  "nitrate_risk_pct",
        "Uranium - Percent of samples U> 30 ppb":          "uranium_risk_pct",
    }
    for src, dst in col_map.items():
        cdf[dst] = pd.to_numeric(cdf[src], errors="coerce").fillna(0.0)
    _state["risk_lookup"] = (
        cdf.set_index("state_name")[list(col_map.values())].to_dict("index")
    )
    print(f"[startup] Risk lookup: {len(_state['risk_lookup'])} states")


def _load_models() -> None:
    qhat_path = SAVED / "jalvaani_adaptive_qhat.pkl"
    if qhat_path.exists():
        _state["adaptive_qhat"] = joblib.load(qhat_path)
        print(f"[startup] Adaptive q_hat: {_state['adaptive_qhat']}")
    else:
        print("[startup] WARNING: jalvaani_adaptive_qhat.pkl missing")

    for name, mod in [
        ("Day1-DepthEnsemble",  depth_model),
        ("Day2-PhysicsNN",      physics_model),
        ("Day3-Contamination",  contamination_model),
        ("Day5-Forecasting",    forecasting_model),
    ]:
        try:
            if mod.load():
                _state["models_loaded"].append(name)
                print(f"[startup] ✓ {name}")
        except Exception as exc:
            print(f"[startup] ✗ {name}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading JalVaani models and lookup data...")
    try:
        _load_station_lookup()
    except Exception as e:
        print(f"[startup] WARNING: station lookup failed: {e}")
    try:
        _load_risk_lookup()
    except Exception as e:
        print(f"[startup] WARNING: risk lookup failed: {e}")
        _state["risk_lookup"] = {}
    _load_models()
    print(f"[startup] Ready. Models: {_state['models_loaded']}")
    yield
    cache.clear()
    print("[shutdown] Cache cleared.")


# ── Rate-limit middleware ─────────────────────────────────────────────────────

class _RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window in-process rate limiter (per IP, per minute).

    For multi-instance deployments replace the in-memory counter with a
    Redis INCR + EXPIRE pattern (atomicity guaranteed by Redis single thread).
    """
    def __init__(self, app, rpm: int = 120) -> None:
        super().__init__(app)
        self._rpm   = rpm
        self._window = 60.0
        self._counts: dict[str, list[float]] = defaultdict(list)
        self._lock   = Lock()

    async def dispatch(self, request: Request, call_next):
        # Skip rate-limiting for static assets
        if request.url.path.startswith("/assets"):
            return await call_next(request)

        ip  = (request.client.host if request.client else "unknown")
        now = time.monotonic()

        with self._lock:
            ts = self._counts[ip]
            # Evict timestamps outside the rolling window
            self._counts[ip] = [t for t in ts if now - t < self._window]
            if len(self._counts[ip]) >= self._rpm:
                return JSONResponse(
                    {"error": "Rate limit exceeded.", "retry_after_seconds": 60},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
            self._counts[ip].append(now)

        return await call_next(request)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="JalVaani AI API",
    description=(
        "**India's Groundwater Intelligence Platform**\n\n"
        "Predict water-table depth, assess fluoride/arsenic/nitrate contamination risk, "
        "and forecast depletion at 16,693 monitoring stations — powered by physics-guided "
        "ML trained on ~903,000 real CGWB/State Board readings (2013–2021).\n\n"
        "**Data source:** Central Ground Water Board (CGWB) and State Groundwater Boards. "
        "Contamination statistics from Rajya Sabha Session 267, AU-1211."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Order matters: GZip → RateLimit → CORS (outer layers wrap inner)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(_RateLimitMiddleware, rpm=settings.rate_limit_rpm)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Internal helpers (synchronous — called inside asyncio.to_thread) ──────────

def _require_station_df():
    if _state["station_df"] is None:
        raise HTTPException(503, "Station lookup not available — ensure jalvaani_real_cleaned.csv is in data/")


def _get_station(lat: float, lon: float) -> dict:
    _require_station_df()
    return utils.find_nearest_station(lat, lon, _state["station_df"])


def _d1_features(lat, lon, date, station) -> np.ndarray:
    return utils.build_feature_vector(
        lat, lon, date, station,
        depth_model.get_feature_names(),
        _state["risk_lookup"] or {},
    )


def _d3_features(lat, lon, date, station) -> np.ndarray:
    return utils.build_feature_vector(
        lat, lon, date, station,
        contamination_model.get_feature_names(),
        _state["risk_lookup"] or {},
    )


def _uncertainty(depth: float) -> dict:
    return utils.get_uncertainty_interval(depth, _state["adaptive_qhat"] or {})


# ── Endpoints — Info ──────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "api": "JalVaani AI",
        "version": "1.0.0",
        "description": "Groundwater intelligence platform for India",
        "data_source": "CGWB + State Groundwater Boards (~903,000 readings, 2013–2021)",
        "endpoints": {
            "POST /predict/depth":        "Groundwater depth prediction with conformal uncertainty",
            "POST /predict/contamination":"Fluoride / arsenic / nitrate risk classification",
            "GET  /forecast/{station}":   "3/6/12-month depletion forecast",
            "POST /report/full":          "Integrated depth + contamination + forecast + physics check",
            "GET  /stations":             "Paginated list of monitoring stations",
            "GET  /stations/search":      "Filter stations by state / district / name",
            "GET  /stats/national":       "National groundwater summary statistics",
            "GET  /health":               "API health and loaded-model list",
        },
    }


@app.get("/health", tags=["Info"])
async def health():
    return {
        "status": "ok",
        "models_loaded": _state["models_loaded"],
        "station_lookup_size": (
            len(_state["station_df"]) if _state["station_df"] is not None else 0
        ),
        "adaptive_qhat_available": _state["adaptive_qhat"] is not None,
        "cache_entries": len(cache),
    }


# ── Endpoints — Prediction ────────────────────────────────────────────────────

@app.post("/predict/depth", response_model=DepthPredictionResponse, tags=["Prediction"])
async def predict_depth(loc: LocationInput):
    """
    Predict groundwater depth (mbgl) at a lat/lon on a given date.

    Uses the Day 1 stacking ensemble (XGBoost + RF → Ridge, R² 0.904).
    Conformal prediction interval from Day 4 (depth-adaptive, 90% target).
    Nearest monitoring station is used for spatial feature engineering.
    """
    def _work():
        station  = _get_station(loc.latitude, loc.longitude)
        features = _d1_features(loc.latitude, loc.longitude, loc.date, station)
        try:
            depth = depth_model.predict(features)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        except Exception as e:
            raise HTTPException(500, f"Prediction error: {e}")
        unc = _uncertainty(depth)
        return DepthPredictionResponse(
            predicted_depth_mbgl=round(depth, 3),
            confidence_interval_90={"lower": unc["lower"], "upper": unc["upper"]},
            uncertainty_level=unc["level"],
            nearest_station=str(station["station_name"]),
            state=str(station["state_name"]),
            district=str(station["district_name"]),
        )

    return await asyncio.to_thread(_work)


@app.post("/predict/contamination", response_model=ContaminationResponse, tags=["Prediction"])
async def predict_contamination(loc: LocationInput):
    """
    Classify fluoride, arsenic, and nitrate contamination risk.

    Probabilities reflect state-level CGWB contamination prevalence (Day 3).
    Fluoride (AUC 0.80 leave-states-out) is the most reliable signal.
    Arsenic and nitrate scores are indicative only — treat with caution.
    """
    def _work():
        station  = _get_station(loc.latitude, loc.longitude)
        features = _d3_features(loc.latitude, loc.longitude, loc.date, station)
        try:
            result = contamination_model.predict(features)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        except Exception as e:
            raise HTTPException(500, f"Prediction error: {e}")
        return ContaminationResponse(
            fluoride_risk=result["fluoride"],
            arsenic_risk=result["arsenic"],
            nitrate_risk=result["nitrate"],
            overall_risk_score=result["overall_risk_score"],
            weighted_avg_risk=result["weighted_avg_risk"],
            risk_drivers=result["risk_drivers"],
        )

    return await asyncio.to_thread(_work)


# ── Endpoints — Forecasting ───────────────────────────────────────────────────

@app.get("/forecast/{station_name}", response_model=ForecastResponse, tags=["Forecasting"])
async def get_forecast(station_name: str):
    """
    3-, 6-, and 12-month groundwater depth forecast for a named station.

    Pre-computed by the Day 5 LSTM (RMSE 3.89 mbgl at 12m).
    Conformal intervals from Day 5 validation calibration (q_hat 2.43/2.77/3.05 mbgl).
    Available for 16,693 CGWB monitoring stations with ≥20 historical readings.
    Use /stations to find valid station names.

    forecast_quality = "low_confidence" when last_observed is outside the
    station's training scaler range (OOD extrapolation) or horizon spread > 5 mbgl.
    38.6% of stations are expected to return low_confidence.
    """
    def _work():
        try:
            result = forecasting_model.forecast(station_name)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        except Exception as e:
            raise HTTPException(500, f"Forecast error: {e}")
        if result is None:
            raise HTTPException(
                404,
                detail={
                    "error": f"Station '{station_name}' not found in forecast data.",
                    "note": "Forecasting covers 16,693 CGWB monitoring stations with ≥20 readings.",
                    "hint": "Use GET /stations or GET /stations/search to browse valid station names.",
                },
            )
        return ForecastResponse(**result)

    return await asyncio.to_thread(_work)


# ── Endpoints — Full report ───────────────────────────────────────────────────

@app.post("/report/full", response_model=FullReportResponse, tags=["Report"])
async def full_report(loc: LocationInput):
    """
    Integrated report: depth prediction + contamination risk + depletion forecast
    + physics consistency check, all for one location in a single request.
    """
    def _work():
        station = _get_station(loc.latitude, loc.longitude)

        # — Depth (Day 1) —
        d1_feat = _d1_features(loc.latitude, loc.longitude, loc.date, station)
        try:
            depth = depth_model.predict(d1_feat)
        except Exception as e:
            raise HTTPException(500, f"Depth prediction failed: {e}")
        unc = _uncertainty(depth)
        depth_resp = DepthPredictionResponse(
            predicted_depth_mbgl=round(depth, 3),
            confidence_interval_90={"lower": unc["lower"], "upper": unc["upper"]},
            uncertainty_level=unc["level"],
            nearest_station=str(station["station_name"]),
            state=str(station["state_name"]),
            district=str(station["district_name"]),
        )

        # — Contamination (Day 3) —
        d3_feat = _d3_features(loc.latitude, loc.longitude, loc.date, station)
        try:
            cont = contamination_model.predict(d3_feat)
        except Exception:
            cont = {
                "fluoride": {"risk_level": "unavailable", "probability": 0.0},
                "arsenic":  {"risk_level": "unavailable", "probability": 0.0},
                "nitrate":  {"risk_level": "unavailable", "probability": 0.0},
                "overall_risk_score": 0.0,
                "weighted_avg_risk":  0.0,
                "risk_drivers": [],
            }
        cont_resp = ContaminationResponse(
            fluoride_risk=cont["fluoride"],
            arsenic_risk=cont["arsenic"],
            nitrate_risk=cont["nitrate"],
            overall_risk_score=cont["overall_risk_score"],
            weighted_avg_risk=cont["weighted_avg_risk"],
            risk_drivers=cont["risk_drivers"],
        )

        # — Forecast (Day 5): exact station → nearest known → flat fallback —
        fc_data = None
        try:
            fc_data = forecasting_model.forecast(str(station["station_name"]))
            if fc_data is None:
                near = forecasting_model.nearest_forecast_stations(
                    loc.latitude, loc.longitude, n=1
                )
                if near:
                    fc_data = forecasting_model.forecast(near[0]["station_name"])
        except Exception:
            pass

        if fc_data is None:
            fc_data = {
                "station_name":       str(station["station_name"]),
                "last_observed_depth": round(depth, 3),
                "forecast_3_month":  {"value": round(depth, 3), "lower": round(max(0.0, depth - 2.43), 3), "upper": round(depth + 2.43, 3)},
                "forecast_6_month":  {"value": round(depth, 3), "lower": round(max(0.0, depth - 2.77), 3), "upper": round(depth + 2.77, 3)},
                "forecast_12_month": {"value": round(depth, 3), "lower": round(max(0.0, depth - 3.05), 3), "upper": round(depth + 3.05, 3)},
                "trend": "unknown",
                "forecast_quality": "low_confidence",
                "forecast_quality_note": "No pre-computed forecast for this station; using current depth prediction as a flat baseline.",
                "horizon_spread_mbgl": 0.0,
            }
        fc_resp = ForecastResponse(**fc_data)

        # — Physics consistency check (Day 2) —
        try:
            phys_depth = physics_model.predict(d1_feat)
            diff = abs(depth - phys_depth)
            phys_check = {
                "ensemble_depth_mbgl":   round(depth, 3),
                "physics_nn_depth_mbgl": round(phys_depth, 3),
                "difference_mbgl":       round(diff, 3),
                "consistent":            diff <= 2.0,
                "note": (
                    "Predictions agree within 2 mbgl."
                    if diff <= 2.0
                    else f"Predictions differ by {diff:.1f} mbgl — "
                         "may indicate unusual hydrogeological conditions at this location."
                ),
            }
        except Exception:
            phys_check = {
                "ensemble_depth_mbgl":   round(depth, 3),
                "physics_nn_depth_mbgl": None,
                "difference_mbgl":       None,
                "consistent":            None,
                "note":                  "Physics model unavailable.",
            }

        summary = (
            f"Near {station['station_name']} in {station['district_name']}, "
            f"{station['state_name']}: predicted depth {round(depth, 1)} mbgl "
            f"(90% CI {unc['lower']}–{unc['upper']} mbgl, {unc['level']} uncertainty). "
            f"Contamination risk — fluoride: {cont['fluoride']['risk_level']}, "
            f"arsenic: {cont['arsenic']['risk_level']}, "
            f"nitrate: {cont['nitrate']['risk_level']}. "
            f"12-month forecast: {fc_resp.forecast_12_month['value']} mbgl, "
            f"trend: {fc_resp.trend}."
        )

        return FullReportResponse(
            location=loc,
            depth_prediction=depth_resp,
            contamination=cont_resp,
            forecast=fc_resp,
            physics_consistency_check=phys_check,
            summary=summary,
        )

    return await asyncio.to_thread(_work)


# ── Endpoints — Stations ──────────────────────────────────────────────────────

@app.get("/stations", tags=["Stations"])
async def list_stations(
    page: int     = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=200, description="Results per page"),
):
    """Paginated list of all CGWB monitoring stations with coordinates."""
    _require_station_df()
    cache_key = f"stations:{page}:{per_page}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    df    = _state["station_df"]
    total = len(df)
    start = (page - 1) * per_page
    chunk = df.iloc[start: start + per_page]
    result = {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page,
        "stations": chunk[
            ["station_name", "state_name", "district_name", "latitude", "longitude"]
        ].to_dict("records"),
    }
    cache.set(cache_key, result, ttl_seconds=settings.cache_ttl_stations_page)
    return result


@app.get("/stations/search", tags=["Stations"])
async def search_stations(
    q:        Optional[str] = Query(None, description="Search across state, district AND station name (OR logic) — e.g. 'kolkata'"),
    state:    Optional[str] = Query(None, description="Filter by state name only"),
    district: Optional[str] = Query(None, description="Filter by district name only"),
    name:     Optional[str] = Query(None, description="Filter by station code only"),
    limit:    int            = Query(20, ge=1, le=100, description="Max results"),
):
    """
    Search stations. Use ?q= for a unified search across state, district and station name (OR).
    Use ?state= / ?district= / ?name= for precise AND-filtered searches.
    """
    _require_station_df()
    df = _state["station_df"]
    if q:
        ql = q.lower()
        mask = (
            df["state_name"].str.lower().str.contains(ql, na=False) |
            df["district_name"].str.lower().str.contains(ql, na=False) |
            df["station_name"].str.lower().str.contains(ql, na=False)
        )
        df = df[mask]
    else:
        if state:
            df = df[df["state_name"].str.lower().str.contains(state.lower(), na=False)]
        if district:
            df = df[df["district_name"].str.lower().str.contains(district.lower(), na=False)]
        if name:
            df = df[df["station_name"].str.lower().str.contains(name.lower(), na=False)]
    rows = df[["station_name", "state_name", "district_name", "latitude", "longitude"]]
    return {
        "count":    len(rows),
        "stations": rows.head(limit).to_dict("records"),
    }


# ── Endpoints — Stats ─────────────────────────────────────────────────────────

@app.get("/stats/national", tags=["Stats"])
async def national_stats():
    """
    National groundwater summary: station counts, depletion trends,
    and contamination prevalence by type.

    Response is cached for 1 hour — data does not change at runtime.
    """
    cached = cache.get("national_stats")
    if cached is not None:
        return cached

    out: dict = {
        "data_source":         "CGWB + State Groundwater Boards, 2013–2021",
        "contamination_source": "Rajya Sabha Session 267, AU-1211 (CGWB state-level statistics)",
    }

    if _state["station_df"] is not None:
        df = _state["station_df"]
        out["total_monitoring_stations"] = int(len(df))
        out["states_covered"]            = int(df["state_name"].nunique())

    out.update(forecasting_model.get_national_stats())

    if _state["risk_lookup"]:
        rl = _state["risk_lookup"]
        fl = [v["fluoride_risk_pct"] for v in rl.values()]
        ar = [v["arsenic_risk_pct"]  for v in rl.values()]
        ni = [v["nitrate_risk_pct"]  for v in rl.values()]
        out["contamination_avg_exceedance_pct"] = {
            "fluoride": round(sum(fl) / len(fl), 2),
            "arsenic":  round(sum(ar) / len(ar), 2),
            "nitrate":  round(sum(ni) / len(ni), 2),
        }

    cache.set("national_stats", out, ttl_seconds=settings.cache_ttl_national_stats)
    return out


# ── Static SPA serving (optional — set STATIC_DIR in .env) ───────────────────
# API routes are registered first; this catch-all only fires if no API route matched.

_static_dir = Path(settings.static_dir).resolve() if settings.static_dir else None

if _static_dir and _static_dir.exists():
    # Serve /assets/* directly (long-lived, Vite fingerprints filenames)
    app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all: return index.html so React Router handles client-side routing."""
        index = _static_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(404, "UI not built. Run: cd jalvaani_ui && npm run build")
