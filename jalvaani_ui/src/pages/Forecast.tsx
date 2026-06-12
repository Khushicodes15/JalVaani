import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { TrendingDown, TrendingUp, Minus, AlertTriangle, CheckCircle } from 'lucide-react'
import { api } from '../api/client'
import { PageShell } from '../components/layout/PageShell'
import { StationSearch } from '../components/forms/StationSearch'
import { Card, StatCard } from '../components/ui/Card'
import { ForecastChart } from '../components/charts/ForecastChart'
import { Badge } from '../components/ui/Badge'
import { Alert } from '../components/ui/Alert'
import { Button } from '../components/ui/Button'
import type { ForecastResponse, TrendDirection } from '../types/api'

const TREND_CONFIG: Record<TrendDirection, { icon: React.ReactNode; label: string; variant: any }> = {
  depleting:  { icon: <TrendingDown size={16} />, label: 'Depleting',  variant: 'depleting'  },
  stable:     { icon: <Minus size={16} />,        label: 'Stable',     variant: 'stable'     },
  recovering: { icon: <TrendingUp size={16} />,   label: 'Recovering', variant: 'recovering' },
  unknown:    { icon: <Minus size={16} />,        label: 'Unknown',    variant: 'unknown'    },
}

export default function Forecast() {
  const [searchParams] = useSearchParams()
  const [stationName, setStationName] = useState(() => searchParams.get('station') ?? '')
  const [result, setResult] = useState<ForecastResponse | null>(null)

  const { mutate, isPending, error } = useMutation({
    mutationFn: () => api.forecast(stationName),
    onSuccess:  (data) => setResult(data),
  })

  // Auto-trigger if station came from URL (e.g. Station Explorer "Forecast" link)
  useEffect(() => {
    const station = searchParams.get('station')
    if (station) {
      setStationName(station)
      mutate()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const r      = result
  const trend  = r ? TREND_CONFIG[r.trend] : null

  return (
    <PageShell
      title="Station Forecast"
      subtitle="3-, 6-, and 12-month groundwater depletion forecasts from the Day 5 LSTM."
    >
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Input */}
        <Card className="xl:col-span-1">
          <h2 className="text-base font-semibold text-well-900 mb-5">Find a Station</h2>
          <div className="space-y-4">
            <StationSearch
              onSelect={(name) => setStationName(name)}
              initialValue={stationName}
            />
            <Button
              onClick={() => stationName && mutate()}
              loading={isPending}
              disabled={!stationName}
              size="lg"
              className="w-full"
            >
              Get Forecast
            </Button>
          </div>
          {error && (
            <Alert variant="error" className="mt-4">
              {(error as Error).message || 'Station not found in forecast dataset.'}
            </Alert>
          )}
        </Card>

        {/* Results */}
        <div className="xl:col-span-2 space-y-5">
          {!r && !isPending && (
            <Card className="flex items-center justify-center h-64 text-well-700/40">
              <div className="text-center">
                <p className="text-4xl mb-3">📈</p>
                <p className="text-sm">Search for a station to view its forecast</p>
              </div>
            </Card>
          )}

          {r && (
            <>
              {/* Header row */}
              <Card className="bg-water-gradient text-white border-0">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <p className="text-well-200 text-xs uppercase tracking-wider mb-1">Station</p>
                    <p className="text-2xl font-bold">{r.station_name}</p>
                    <p className="text-well-300 text-sm mt-1">
                      Last observed: <strong className="text-white">{r.last_observed_depth} mbgl</strong>
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    {trend && (
                      <Badge variant={trend.variant} dot className="text-sm px-3 py-1">
                        {trend.icon}
                        {trend.label}
                      </Badge>
                    )}
                    <Badge variant={r.forecast_quality}>
                      {r.forecast_quality === 'reliable'
                        ? <><CheckCircle size={11} className="mr-1" />Reliable</>
                        : <><AlertTriangle size={11} className="mr-1" />Low Confidence</>
                      }
                    </Badge>
                  </div>
                </div>

                {r.forecast_quality === 'low_confidence' && (
                  <div className="mt-4 p-3 bg-amber-500/20 border border-amber-400/30 rounded-xl text-xs text-amber-100">
                    ⚠ {r.forecast_quality_note}
                  </div>
                )}
              </Card>

              {/* Chart */}
              <Card>
                <h2 className="text-sm font-semibold text-well-900 mb-4">
                  3 / 6 / 12-Month Forecast
                  <span className="text-xs font-normal text-well-700/50 ml-2">
                    Y-axis inverted — deeper = lower
                  </span>
                </h2>
                <ForecastChart data={r} />
                <p className="text-xs text-well-700/40 mt-3">
                  Shaded band = 90% conformal interval (q̂ 2.43 / 2.77 / 3.05 mbgl).
                  Horizon spread: {r.horizon_spread_mbgl.toFixed(1)} mbgl.
                </p>
              </Card>

              {/* Horizon cards */}
              <div className="grid grid-cols-3 gap-4">
                {([
                  { label: '3-Month',  fc: r.forecast_3_month  },
                  { label: '6-Month',  fc: r.forecast_6_month  },
                  { label: '12-Month', fc: r.forecast_12_month },
                ] as const).map(({ label, fc }) => (
                  <Card key={label} padding="sm">
                    <p className="text-xs text-well-700/50 font-medium mb-2">{label}</p>
                    <p className="text-xl font-bold text-well-900 tabular-nums">{fc.value} m</p>
                    <p className="text-xs text-well-700/40 mt-1">
                      {fc.lower}–{fc.upper} m
                    </p>
                  </Card>
                ))}
              </div>

              <Alert variant="info">
                LSTM forecast (RMSE 3.89 mbgl at 12m, R² 0.879). Conformal intervals calibrated
                on validation sequences; actual test coverage 82–84% (temporal distribution shift).
                38.6% of stations return low_confidence due to out-of-distribution inputs.
              </Alert>
            </>
          )}
        </div>
      </div>
    </PageShell>
  )
}
