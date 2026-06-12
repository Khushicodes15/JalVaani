// TypeScript interfaces mirroring every FastAPI Pydantic schema exactly.
// Keep in sync with jalvaani_api/schemas.py.

export interface LocationInput {
  latitude: number   // 6.0–38.0 (India bounding box)
  longitude: number  // 68.0–98.0
  date: string       // "YYYY-MM-DD"
}

export interface ConfidenceInterval {
  lower: number
  upper: number
}

export interface DepthPredictionResponse {
  predicted_depth_mbgl: number
  confidence_interval_90: ConfidenceInterval
  uncertainty_level: 'low' | 'medium' | 'high'
  nearest_station: string
  state: string
  district: string
}

export interface RiskDetail {
  risk_level: 'low' | 'medium' | 'high' | 'unavailable'
  probability: number
}

export interface ContaminationResponse {
  fluoride_risk: RiskDetail
  arsenic_risk: RiskDetail
  nitrate_risk: RiskDetail
  overall_risk_score: number    // max of the three probabilities
  weighted_avg_risk: number     // 0.4·F + 0.4·A + 0.2·N
  risk_drivers: string[]        // contaminants classified as "high"
}

export interface ForecastInterval {
  value: number
  lower: number
  upper: number
}

export type ForecastQuality = 'reliable' | 'low_confidence'
export type TrendDirection = 'depleting' | 'stable' | 'recovering' | 'unknown'

export interface ForecastResponse {
  station_name: string
  last_observed_depth: number
  forecast_3_month: ForecastInterval
  forecast_6_month: ForecastInterval
  forecast_12_month: ForecastInterval
  trend: TrendDirection
  forecast_quality: ForecastQuality
  forecast_quality_note: string
  horizon_spread_mbgl: number
}

export interface PhysicsCheck {
  ensemble_depth_mbgl: number
  physics_nn_depth_mbgl: number | null
  difference_mbgl: number | null
  consistent: boolean | null
  note: string
}

export interface FullReportResponse {
  location: LocationInput
  depth_prediction: DepthPredictionResponse
  contamination: ContaminationResponse
  forecast: ForecastResponse
  physics_consistency_check: PhysicsCheck
  summary: string
}

// /stations
export interface Station {
  station_name: string
  state_name: string
  district_name: string
  latitude: number
  longitude: number
}

export interface StationListResponse {
  total: number
  page: number
  per_page: number
  pages: number
  stations: Station[]
}

export interface StationSearchResponse {
  count: number
  stations: Station[]
}

// /stats/national
export interface NationalStats {
  data_source: string
  contamination_source: string
  total_monitoring_stations: number
  states_covered: number
  forecasted_stations: number
  stations_depleting: number
  depletion_pct: number
  top_5_depleting_states: Record<string, number>
  contamination_avg_exceedance_pct: {
    fluoride: number
    arsenic: number
    nitrate: number
  }
}

// /health
export interface HealthResponse {
  status: string
  models_loaded: string[]
  station_lookup_size: number
  adaptive_qhat_available: boolean
  cache_entries: number
}
