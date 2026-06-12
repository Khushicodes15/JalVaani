import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { CheckCircle, XCircle, FileText } from 'lucide-react'
import { api } from '../api/client'
import { PageShell } from '../components/layout/PageShell'
import { LocationForm } from '../components/forms/LocationForm'
import { Card } from '../components/ui/Card'
import { RiskGauge } from '../components/charts/RiskGauge'
import { ForecastChart } from '../components/charts/ForecastChart'
import { Badge } from '../components/ui/Badge'
import { Alert } from '../components/ui/Alert'
import type { FullReportResponse, LocationInput } from '../types/api'
import { clsx } from 'clsx'

export default function FullReport() {
  const [result, setResult] = useState<FullReportResponse | null>(null)

  const { mutate, isPending, error } = useMutation({
    mutationFn: api.fullReport,
    onSuccess: (data) => setResult(data),
  })

  const r = result
  const dp = r?.depth_prediction
  const co = r?.contamination
  const fc = r?.forecast
  const ph = r?.physics_consistency_check

  return (
    <PageShell
      title="Full Report"
      subtitle="Integrated depth, contamination, forecast, and physics consistency for one location."
    >
      <div className="space-y-6">
        {/* Input */}
        <Card>
          <h2 className="text-base font-semibold text-well-900 mb-5">Location & Date</h2>
          <LocationForm
            onSubmit={(loc: LocationInput) => mutate(loc)}
            loading={isPending}
            submitLabel="Generate Full Report"
          />
          {error && (
            <Alert variant="error" className="mt-4">
              {(error as Error).message || 'Report generation failed.'}
            </Alert>
          )}
        </Card>

        {!r && !isPending && (
          <Card className="flex items-center justify-center h-40 text-well-700/40">
            <div className="text-center">
              <FileText className="w-10 h-10 mx-auto mb-3 opacity-40" />
              <p className="text-sm">Your full report will appear here</p>
            </div>
          </Card>
        )}

        {r && dp && co && fc && ph && (
          <>
            {/* Summary banner */}
            <Card className="bg-water-gradient text-white border-0">
              <p className="text-well-200 text-xs uppercase tracking-wider mb-2">Summary</p>
              <p className="text-sm leading-relaxed">{r.summary}</p>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Depth */}
              <Card>
                <h2 className="text-sm font-semibold text-well-900 mb-4">
                  Depth Prediction · Day 1 Ensemble
                </h2>
                <div className="flex items-baseline gap-2 mb-2">
                  <span className="text-4xl font-bold text-well-700 tabular-nums">
                    {dp.predicted_depth_mbgl}
                  </span>
                  <span className="text-well-700/60">mbgl</span>
                  <Badge
                    variant={dp.uncertainty_level === 'low' ? 'safe' : dp.uncertainty_level === 'medium' ? 'medium' : 'high'}
                    className="ml-auto"
                  >
                    {dp.uncertainty_level} uncertainty
                  </Badge>
                </div>
                <p className="text-sm text-well-700/60">
                  90% CI: {dp.confidence_interval_90.lower}–{dp.confidence_interval_90.upper} m
                  · Station: {dp.nearest_station}
                </p>
                <p className="text-sm text-well-700/60 mt-1">
                  {dp.district}, {dp.state}
                </p>
              </Card>

              {/* Physics check */}
              <Card>
                <h2 className="text-sm font-semibold text-well-900 mb-4">
                  Physics Consistency · Day 1 vs Day 2
                </h2>
                {ph.consistent !== null ? (
                  <div className={clsx(
                    'flex items-center gap-3 p-3 rounded-xl mb-3',
                    ph.consistent ? 'bg-wetland-50 border border-wetland-200' : 'bg-amber-50 border border-amber-200'
                  )}>
                    {ph.consistent
                      ? <CheckCircle className="text-wetland-600 w-5 h-5 flex-shrink-0" />
                      : <XCircle    className="text-amber-600  w-5 h-5 flex-shrink-0" />}
                    <p className={clsx('text-sm font-medium', ph.consistent ? 'text-wetland-700' : 'text-amber-700')}>
                      {ph.note}
                    </p>
                  </div>
                ) : null}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-well-700/50 text-xs">Ensemble (Day 1)</p>
                    <p className="font-semibold text-well-900">{ph.ensemble_depth_mbgl} mbgl</p>
                  </div>
                  <div>
                    <p className="text-well-700/50 text-xs">GroundwaterNet (Day 2)</p>
                    <p className="font-semibold text-well-900">
                      {ph.physics_nn_depth_mbgl ?? 'N/A'} {ph.physics_nn_depth_mbgl ? 'mbgl' : ''}
                    </p>
                  </div>
                  {ph.difference_mbgl !== null && (
                    <div className="col-span-2">
                      <p className="text-well-700/50 text-xs">Difference</p>
                      <p className="font-semibold text-well-900">{ph.difference_mbgl} mbgl</p>
                    </div>
                  )}
                </div>
              </Card>
            </div>

            {/* Contamination */}
            <Card>
              <div className="flex items-center justify-between mb-5">
                <h2 className="text-sm font-semibold text-well-900">
                  Contamination Risk · Day 3 Classifiers
                </h2>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-well-700/50">Overall score:</span>
                  <span className={clsx('text-xl font-bold tabular-nums', {
                    'text-danger-600':  co.overall_risk_score >= 0.6,
                    'text-amber-600':   co.overall_risk_score >= 0.35 && co.overall_risk_score < 0.6,
                    'text-wetland-600': co.overall_risk_score < 0.35,
                  })}>
                    {Math.round(co.overall_risk_score * 100)}%
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <RiskGauge label="Fluoride"  data={co.fluoride_risk} />
                <RiskGauge label="Arsenic"   data={co.arsenic_risk}  />
                <RiskGauge label="Nitrate"   data={co.nitrate_risk}  />
              </div>
              {co.risk_drivers.length > 0 && (
                <div className="flex items-center gap-2 mt-4">
                  <span className="text-xs text-well-700/50">High-risk contaminants:</span>
                  {co.risk_drivers.map((d) => (
                    <Badge key={d} variant="high" dot>{d}</Badge>
                  ))}
                </div>
              )}
            </Card>

            {/* Forecast */}
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-well-900">
                  Depletion Forecast · Day 5 LSTM · {fc.station_name}
                </h2>
                <Badge variant={fc.trend as any} dot>{fc.trend}</Badge>
              </div>
              <ForecastChart data={fc} />
              {fc.forecast_quality === 'low_confidence' && (
                <Alert variant="warning" className="mt-4">
                  {fc.forecast_quality_note}
                </Alert>
              )}
            </Card>
          </>
        )}
      </div>
    </PageShell>
  )
}
