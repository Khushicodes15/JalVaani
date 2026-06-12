import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { MapPin, Layers, AlertCircle } from 'lucide-react'
import { api } from '../api/client'
import { PageShell } from '../components/layout/PageShell'
import { LocationForm } from '../components/forms/LocationForm'
import { Card, StatCard } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { Alert } from '../components/ui/Alert'
import type { DepthPredictionResponse, LocationInput } from '../types/api'

export default function DepthPredictor() {
  const [result, setResult] = useState<DepthPredictionResponse | null>(null)

  const { mutate, isPending, error } = useMutation({
    mutationFn: api.predictDepth,
    onSuccess: (data) => setResult(data),
  })

  const d = result

  // CI visualization width
  const ciWidth = d
    ? d.confidence_interval_90.upper - d.confidence_interval_90.lower
    : 0
  const maxDepth = d ? Math.max(d.confidence_interval_90.upper, 50) : 50
  const loPct    = d ? (d.confidence_interval_90.lower / maxDepth) * 100 : 0
  const hiPct    = d ? (d.confidence_interval_90.upper / maxDepth) * 100 : 0
  const midPct   = d ? (d.predicted_depth_mbgl / maxDepth) * 100 : 0

  return (
    <PageShell
      title="Depth Predictor"
      subtitle="Predict groundwater depth (mbgl) at any Indian location using the Day 1 stacking ensemble."
    >
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Input card */}
        <Card className="xl:col-span-1">
          <h2 className="text-base font-semibold text-well-900 mb-5">Location & Date</h2>
          <LocationForm
            onSubmit={(loc: LocationInput) => mutate(loc)}
            loading={isPending}
            submitLabel="Predict Depth"
          />
          {error && (
            <Alert variant="error" className="mt-4">
              {(error as Error).message || 'Prediction failed. Check the server is running.'}
            </Alert>
          )}
        </Card>

        {/* Results */}
        <div className="xl:col-span-2 space-y-4">
          {!d && !isPending && (
            <Card className="flex items-center justify-center h-48 text-well-700/40">
              <div className="text-center">
                <Layers className="w-10 h-10 mx-auto mb-3 opacity-40" />
                <p className="text-sm">Fill in the form and click Predict Depth</p>
              </div>
            </Card>
          )}

          {d && (
            <>
              {/* Main depth result */}
              <Card className="bg-water-gradient text-white border-0">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-well-200 text-xs uppercase tracking-wider mb-1">
                      Predicted Groundwater Depth
                    </p>
                    <p className="text-5xl font-bold tabular-nums">
                      {d.predicted_depth_mbgl}
                      <span className="text-2xl ml-1 font-normal opacity-80"> mbgl</span>
                    </p>
                    <p className="text-well-300 text-sm mt-2">
                      Near {d.nearest_station} · {d.district}, {d.state}
                    </p>
                  </div>
                  <Badge
                    variant={d.uncertainty_level === 'low' ? 'safe' : d.uncertainty_level === 'medium' ? 'medium' : 'high'}
                    className="flex-shrink-0"
                  >
                    {d.uncertainty_level} uncertainty
                  </Badge>
                </div>

                {/* CI bar */}
                <div className="mt-6">
                  <p className="text-well-300 text-xs mb-2">90% Conformal Prediction Interval</p>
                  <div className="relative h-6 bg-white/10 rounded-full overflow-hidden">
                    {/* CI band */}
                    <div
                      className="absolute top-0 bottom-0 bg-white/20 rounded-full"
                      style={{ left: `${loPct}%`, width: `${hiPct - loPct}%` }}
                    />
                    {/* Point estimate */}
                    <div
                      className="absolute top-1 bottom-1 w-1 bg-soil-400 rounded-full"
                      style={{ left: `${midPct}%`, transform: 'translateX(-50%)' }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-well-300 mt-1.5">
                    <span>{d.confidence_interval_90.lower} m</span>
                    <span className="font-medium text-white">{d.predicted_depth_mbgl} m (estimate)</span>
                    <span>{d.confidence_interval_90.upper} m</span>
                  </div>
                  <p className="text-well-400 text-xs mt-1">
                    CI width: {ciWidth.toFixed(1)} m · depth bin-adaptive q̂ (Day 4 conformal calibration)
                  </p>
                </div>
              </Card>

              {/* Station & model info */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                <StatCard label="Nearest Station" value={d.nearest_station} icon={<MapPin size={16} />} accent="blue" />
                <StatCard label="State" value={d.state} />
                <StatCard label="District" value={d.district} />
              </div>

              <Alert variant="info">
                Depth is predicted using the nearest CGWB station's historical median as a proxy
                for <code className="text-xs bg-well-100 px-1 rounded">level_diff_lag</code>.
                Accuracy degrades for locations &gt;50 km from any monitoring station.
                Model: Stacking Ensemble (XGBoost + RF → Ridge), R² 0.904.
              </Alert>
            </>
          )}
        </div>
      </div>
    </PageShell>
  )
}
