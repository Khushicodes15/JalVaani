import { clsx } from 'clsx'
import type { RiskDetail } from '../../types/api'

interface Props {
  label: string
  data: RiskDetail
  note?: string
}

function riskColor(level: string) {
  if (level === 'high')   return { stroke: '#C0392B', text: 'text-danger-600', bg: 'bg-danger-100' }
  if (level === 'medium') return { stroke: '#D97706', text: 'text-amber-600',  bg: 'bg-amber-100'  }
  return                         { stroke: '#1E7A4B', text: 'text-wetland-600', bg: 'bg-wetland-100' }
}

export function RiskGauge({ label, data, note }: Props) {
  const pct    = Math.round(data.probability * 100)
  const radius = 42
  const circ   = 2 * Math.PI * radius
  const dash   = (pct / 100) * circ
  const { stroke, text, bg } = riskColor(data.risk_level)

  return (
    <div className={clsx('rounded-2xl p-5 border border-well-100 bg-white shadow-card text-center', bg, 'bg-opacity-30')}>
      {/* SVG Dial */}
      <div className="relative w-28 h-28 mx-auto">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          {/* Track */}
          <circle cx="50" cy="50" r={radius} fill="none" stroke="#E5E7EB" strokeWidth="9" />
          {/* Progress */}
          <circle
            cx="50" cy="50" r={radius}
            fill="none"
            stroke={stroke}
            strokeWidth="9"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circ}`}
            className="transition-all duration-700"
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={clsx('text-2xl font-bold tabular-nums', text)}>{pct}%</span>
        </div>
      </div>

      <p className="mt-3 text-sm font-semibold text-well-900">{label}</p>
      <span className={clsx(
        'mt-1 inline-block px-2.5 py-0.5 rounded-full text-xs font-medium capitalize',
        data.risk_level === 'high'   && 'bg-danger-100 text-danger-600',
        data.risk_level === 'medium' && 'bg-amber-100 text-amber-600',
        data.risk_level === 'low'    && 'bg-wetland-100 text-wetland-600',
      )}>
        {data.risk_level} risk
      </span>
      {note && <p className="mt-2 text-xs text-well-700/50 text-balance">{note}</p>}
    </div>
  )
}
