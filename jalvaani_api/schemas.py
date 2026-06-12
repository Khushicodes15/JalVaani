"""
JalVaani AI API — Pydantic request/response schemas.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator


class LocationInput(BaseModel):
    latitude: float
    longitude: float
    date: str  # "YYYY-MM-DD"

    @validator("latitude")
    def check_lat(cls, v):
        if not 6.0 <= v <= 38.0:
            raise ValueError("latitude must be 6.0–38.0 (India bounding box)")
        return v

    @validator("longitude")
    def check_lon(cls, v):
        if not 68.0 <= v <= 98.0:
            raise ValueError("longitude must be 68.0–98.0 (India bounding box)")
        return v

    @validator("date")
    def check_date(cls, v):
        try:
            dt = datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be YYYY-MM-DD")
        if not 2000 <= dt.year <= 2030:
            raise ValueError("year must be 2000–2030")
        return v


class DepthPredictionResponse(BaseModel):
    predicted_depth_mbgl: float
    confidence_interval_90: Dict[str, float]   # {"lower": x, "upper": y}
    uncertainty_level: str                      # "low" / "medium" / "high"
    nearest_station: str
    state: str
    district: str


class ContaminationResponse(BaseModel):
    fluoride_risk: Dict[str, Any]    # {"risk_level": str, "probability": float}
    arsenic_risk: Dict[str, Any]
    nitrate_risk: Dict[str, Any]
    overall_risk_score: float        # max of the three probabilities
    weighted_avg_risk: float         # 0.4·fluoride + 0.4·arsenic + 0.2·nitrate
    risk_drivers: List[str]          # contaminants classified as "high"


class ForecastResponse(BaseModel):
    station_name: str
    last_observed_depth: float
    forecast_3_month: Dict[str, float]   # {"value": x, "lower": y, "upper": z}
    forecast_6_month: Dict[str, float]
    forecast_12_month: Dict[str, float]
    trend: str                           # "depleting" / "stable" / "recovering"
    forecast_quality: str                # "reliable" / "low_confidence"
    forecast_quality_note: str           # explanation of quality flag
    horizon_spread_mbgl: float           # max−min across 3m/6m/12m forecasts


class FullReportResponse(BaseModel):
    location: LocationInput
    depth_prediction: DepthPredictionResponse
    contamination: ContaminationResponse
    forecast: ForecastResponse
    physics_consistency_check: Dict[str, Any]
    summary: str
