import { useQuery } from '@tanstack/react-query'
import { Activity, Droplets, MapPin, TrendingDown, AlertTriangle } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { Card, StatCard } from '../components/ui/Card'
import { DepletionBar } from '../components/charts/DepletionBar'
import { FullPageSpinner } from '../components/ui/Spinner'
import { Alert } from '../components/ui/Alert'

export default function Dashboard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['national-stats'],
    queryFn: api.nationalStats,
    staleTime: 60 * 60 * 1000,
  })

  if (isLoading) return <FullPageSpinner label="Loading national statistics…" />
  if (error || !data) return (
    <div className="p-8">
      <Alert variant="error" title="Could not load statistics">
        Make sure the JalVaani API is running on port 8000.
      </Alert>
    </div>
  )

  const depletionPct = data.depletion_pct ?? 0
  const cont         = data.contamination_avg_exceedance_pct

  return (
    <div className="flex-1 animate-fade-in">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <div className="bg-water-gradient text-white relative overflow-hidden">
        {/* Decorative ripple rings */}
        <div className="absolute right-[-60px] top-[-60px] w-64 h-64 opacity-10">
          {[0, 1, 2].map((i) => (
            <span key={i} className="ripple-ring absolute inset-0"
              style={{ animationDelay: `${i * 0.8}s` }} />
          ))}
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 relative">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-full bg-white/15 flex items-center justify-center">
              <Droplets size={20} className="text-white" />
            </div>
            <span className="text-well-200 text-sm font-medium tracking-widest uppercase">
              JalVaani AI — Groundwater Intelligence
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold leading-tight text-balance mb-3">
            India's Groundwater at a Glance
          </h1>
          <p className="text-well-300 text-sm sm:text-base max-w-2xl">
            Real-time predictions powered by ~903,000 CGWB/State Board readings across
            33,921 monitoring stations (2013–2021). Physics-guided ML · Conformal uncertainty.
          </p>

          <div className="flex flex-wrap gap-3 mt-6">
            {[
              { to: '/depth',         label: 'Predict Depth',      icon: '📍' },
              { to: '/contamination', label: 'Check Contamination', icon: '🧪' },
              { to: '/forecast',      label: 'Forecast Station',    icon: '📈' },
            ].map(({ to, label, icon }) => (
              <Link key={to} to={to}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white/15
                           hover:bg-white/25 border border-white/20 text-sm font-medium
                           transition-all backdrop-blur-sm">
                <span>{icon}</span>{label}
              </Link>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* ── Stat cards ───────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Monitoring Stations"
            value={(data.total_monitoring_stations ?? 0).toLocaleString()}
            sub={`${data.states_covered} states & UTs`}
            icon={<MapPin size={18} />}
            accent="blue"
          />
          <StatCard
            label="Forecasted Stations"
            value={(data.forecasted_stations ?? 0).toLocaleString()}
            sub="LSTM 3/6/12-month"
            icon={<Activity size={18} />}
            accent="earth"
          />
          <StatCard
            label="Depleting Stations"
            value={`${depletionPct}%`}
            sub={`${(data.stations_depleting ?? 0).toLocaleString()} of ${(data.forecasted_stations ?? 0).toLocaleString()}`}
            icon={<TrendingDown size={18} />}
            accent="red"
          />
          <StatCard
            label="Avg Fluoride Exceedance"
            value={`${cont?.fluoride ?? 0}%`}
            sub="across all states"
            icon={<AlertTriangle size={18} />}
            accent="earth"
          />
        </div>

        {/* ── Contamination summary ────────────────────────────────────── */}
        {cont && (
          <Card>
            <h2 className="text-base font-semibold text-well-900 mb-4">
              Average Contamination Exceedance by Type
              <span className="text-xs font-normal text-well-700/50 ml-2">(state-level CGWB averages)</span>
            </h2>
            <div className="grid grid-cols-3 gap-4">
              {[
                { label: 'Fluoride (>1.5 mg/L)',  value: cont.fluoride,  color: 'bg-danger-500' },
                { label: 'Nitrate (>45 mg/L)',    value: cont.nitrate,   color: 'bg-amber-500'  },
                { label: 'Arsenic (>10 ppb)',     value: cont.arsenic,   color: 'bg-soil-600'   },
              ].map(({ label, value, color }) => (
                <div key={label} className="text-center">
                  <div className="relative h-2 bg-well-100 rounded-full overflow-hidden mb-2">
                    <div
                      className={`absolute inset-y-0 left-0 rounded-full ${color} transition-all duration-700`}
                      style={{ width: `${Math.min(value * 5, 100)}%` }}
                    />
                  </div>
                  <p className="text-lg font-bold text-well-900">{value}%</p>
                  <p className="text-xs text-well-700/60 text-balance">{label}</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-well-700/40 mt-4">
              Source: Rajya Sabha Session 267, AU-1211 (CGWB state-level statistics)
            </p>
          </Card>
        )}

        {/* ── Depletion chart ──────────────────────────────────────────── */}
        {data.top_5_depleting_states && (
          <Card>
            <h2 className="text-base font-semibold text-well-900 mb-6">
              Top 5 Depleting States
              <span className="text-xs font-normal text-well-700/50 ml-2">
                (fraction of forecasted stations with depleting trend)
              </span>
            </h2>
            <DepletionBar data={data.top_5_depleting_states} />
          </Card>
        )}

        {/* ── Data note ────────────────────────────────────────────────── */}
        <Alert variant="info" title="Research platform — caveats apply">
          Contamination scores reflect state-level CGWB exceedance statistics.
          Fluoride is the most reliable signal (leave-states-out AUC 0.80).
          Arsenic and nitrate scores are indicative only due to state-aggregate labels.
          Forecast intervals are calibrated on validation sequences; test coverage is 82–84%
          (temporal distribution shift caveat). Do not use for drinking-water decisions without
          well-level chemical testing.
        </Alert>
      </div>
    </div>
  )
}
