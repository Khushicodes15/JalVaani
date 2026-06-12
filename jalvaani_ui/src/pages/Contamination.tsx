import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { FlaskConical } from 'lucide-react'
import { api } from '../api/client'
import { PageShell } from '../components/layout/PageShell'
import { LocationForm } from '../components/forms/LocationForm'
import { Card } from '../components/ui/Card'
import { RiskGauge } from '../components/charts/RiskGauge'
import { Badge } from '../components/ui/Badge'
import { Alert } from '../components/ui/Alert'
import type { ContaminationResponse, LocationInput } from '../types/api'
import { clsx } from 'clsx'

const CONTAMINANT_NOTES = {
  fluoride: 'AUC 0.80 leave-states-out — most reliable signal.',
  arsenic:  'AUC 0.41 — state-aggregate labels; treat with caution.',
  nitrate:  'AUC 0.50 — indicative only; well-level testing recommended.',
}

export default function Contamination() {
  const [result, setResult] = useState<ContaminationResponse | null>(null)

  const { mutate, isPending, error } = useMutation({
    mutationFn: api.predictContamination,
    onSuccess: (data) => setResult(data),
  })

  const r = result

  return (
    <PageShell
      title="Contamination Risk"
      subtitle="Fluoride, arsenic, and nitrate risk classification using state-level CGWB statistics."
    >
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Input */}
        <Card className="xl:col-span-1">
          <h2 className="text-base font-semibold text-well-900 mb-5">Location & Date</h2>
          <LocationForm
            onSubmit={(loc: LocationInput) => mutate(loc)}
            loading={isPending}
            submitLabel="Assess Risk"
          />
          {error && (
            <Alert variant="error" className="mt-4">
              {(error as Error).message || 'Request failed. Check server.'}
            </Alert>
          )}
        </Card>

        {/* Results */}
        <div className="xl:col-span-2 space-y-5">
          {!r && !isPending && (
            <Card className="flex items-center justify-center h-48 text-well-700/40">
              <div className="text-center">
                <FlaskConical className="w-10 h-10 mx-auto mb-3 opacity-40" />
                <p className="text-sm">Fill in the form to assess contamination risk</p>
              </div>
            </Card>
          )}

          {r && (
            <>
              {/* Overall score */}
              <Card className="flex items-center justify-between gap-6 flex-wrap">
                <div>
                  <p className="text-xs uppercase tracking-wider text-well-700/50 mb-1">Overall Risk Score</p>
                  <p className={clsx('text-5xl font-bold tabular-nums', {
                    'text-danger-600':  r.overall_risk_score >= 0.6,
                    'text-amber-600':   r.overall_risk_score >= 0.35 && r.overall_risk_score < 0.6,
                    'text-wetland-600': r.overall_risk_score < 0.35,
                  })}>
                    {Math.round(r.overall_risk_score * 100)}%
                  </p>
                  <p className="text-xs text-well-700/50 mt-1">
                    max(F, As, NO₃) · weighted avg: {Math.round(r.weighted_avg_risk * 100)}%
                  </p>
                </div>
                <div className="flex flex-col gap-2 items-end">
                  {r.risk_drivers.length > 0 ? (
                    <>
                      <p className="text-xs font-medium text-well-700/60">Risk drivers</p>
                      <div className="flex flex-wrap gap-2 justify-end">
                        {r.risk_drivers.map((d) => (
                          <Badge key={d} variant="high" dot>
                            {d.charAt(0).toUpperCase() + d.slice(1)}
                          </Badge>
                        ))}
                      </div>
                    </>
                  ) : (
                    <Badge variant="safe" dot>No high-risk contaminants</Badge>
                  )}
                </div>
              </Card>

              {/* Three gauges */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <RiskGauge
                  label="Fluoride"
                  data={r.fluoride_risk}
                  note={CONTAMINANT_NOTES.fluoride}
                />
                <RiskGauge
                  label="Arsenic"
                  data={r.arsenic_risk}
                  note={CONTAMINANT_NOTES.arsenic}
                />
                <RiskGauge
                  label="Nitrate"
                  data={r.nitrate_risk}
                  note={CONTAMINANT_NOTES.nitrate}
                />
              </div>

              <Alert variant="warning" title="Interpretation caution">
                Probabilities reflect <strong>state-level</strong> contamination prevalence, not
                well-level chemistry. Never use these scores as a substitute for actual chemical
                testing. Fluoride is the most spatially predictive contaminant in this dataset.
              </Alert>
            </>
          )}
        </div>
      </div>
    </PageShell>
  )
}
