/**
 * JalVaani API client.
 *
 * Axios instance + typed wrappers for every endpoint.
 * In development, Vite proxies all these paths to http://localhost:8000.
 * In production, requests go to the same origin (FastAPI serves everything).
 */
import axios from 'axios'
import type {
  ContaminationResponse,
  DepthPredictionResponse,
  ForecastResponse,
  FullReportResponse,
  HealthResponse,
  LocationInput,
  NationalStats,
  StationListResponse,
  StationSearchResponse,
} from '../types/api'

const http = axios.create({
  baseURL: '/',
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
})

// Global error normalisation — surfaces FastAPI detail strings cleanly
http.interceptors.response.use(
  (r) => r,
  (err) => {
    const detail = err?.response?.data?.detail
    if (typeof detail === 'string') err.message = detail
    else if (detail?.error) err.message = detail.error
    return Promise.reject(err)
  }
)

// ── Typed API calls ────────────────────────────────────────────────────────────

export const api = {
  health: (): Promise<HealthResponse> =>
    http.get('/health').then((r) => r.data),

  nationalStats: (): Promise<NationalStats> =>
    http.get('/stats/national').then((r) => r.data),

  predictDepth: (loc: LocationInput): Promise<DepthPredictionResponse> =>
    http.post('/predict/depth', loc).then((r) => r.data),

  predictContamination: (loc: LocationInput): Promise<ContaminationResponse> =>
    http.post('/predict/contamination', loc).then((r) => r.data),

  forecast: (stationName: string): Promise<ForecastResponse> =>
    http.get(`/forecast/${encodeURIComponent(stationName)}`).then((r) => r.data),

  fullReport: (loc: LocationInput): Promise<FullReportResponse> =>
    http.post('/report/full', loc).then((r) => r.data),

  listStations: (page: number, perPage = 50): Promise<StationListResponse> =>
    http.get('/stations', { params: { page, per_page: perPage } }).then((r) => r.data),

  searchStations: (params: {
    q?: string
    state?: string
    district?: string
    name?: string
    limit?: number
  }): Promise<StationSearchResponse> =>
    http.get('/stations/search', { params }).then((r) => r.data),
}
