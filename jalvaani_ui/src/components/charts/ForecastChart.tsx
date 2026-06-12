import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'
import type { ForecastResponse } from '../../types/api'

interface Props {
  data: ForecastResponse
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-well-200 rounded-xl shadow-well px-4 py-3 text-sm">
      <p className="font-semibold text-well-900 mb-2">{label}</p>
      {payload.map((p: any) => (
        p.name !== 'CI Band' && (
          <div key={p.name} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: p.color }} />
            <span className="text-well-700/70">{p.name}:</span>
            <span className="font-medium text-well-900">{p.value?.toFixed(2)} mbgl</span>
          </div>
        )
      ))}
    </div>
  )
}

export function ForecastChart({ data }: Props) {
  const chartData = [
    {
      label: 'Last Observed',
      depth: data.last_observed_depth,
      lower: data.last_observed_depth,
      upper: data.last_observed_depth,
      band:  0,
    },
    {
      label: '3 months',
      depth: data.forecast_3_month.value,
      lower: data.forecast_3_month.lower,
      upper: data.forecast_3_month.upper,
      band:  data.forecast_3_month.upper - data.forecast_3_month.lower,
    },
    {
      label: '6 months',
      depth: data.forecast_6_month.value,
      lower: data.forecast_6_month.lower,
      upper: data.forecast_6_month.upper,
      band:  data.forecast_6_month.upper - data.forecast_6_month.lower,
    },
    {
      label: '12 months',
      depth: data.forecast_12_month.value,
      lower: data.forecast_12_month.lower,
      upper: data.forecast_12_month.upper,
      band:  data.forecast_12_month.upper - data.forecast_12_month.lower,
    },
  ]

  // Invert Y-axis: deeper = larger number = lower on screen (standard groundwater convention)
  const allValues = chartData.flatMap((d) => [d.lower, d.upper])
  const yMin = Math.floor(Math.min(...allValues) - 2)
  const yMax = Math.ceil(Math.max(...allValues) + 2)

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#D6EAF8" />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 12, fill: '#1B4F72', opacity: 0.7 }}
          axisLine={{ stroke: '#AED6F1' }}
          tickLine={false}
        />
        <YAxis
          reversed
          domain={[yMin, yMax]}
          tick={{ fontSize: 12, fill: '#1B4F72', opacity: 0.7 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v) => `${v}m`}
          label={{ value: 'Depth (mbgl)', angle: -90, position: 'insideLeft', offset: 12, fontSize: 11, fill: '#1B4F72', opacity: 0.5 }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          formatter={(value) => <span className="text-xs text-well-700/70">{value}</span>}
        />

        {/* Confidence band (shaded area between lower and upper) */}
        <Area
          type="monotone"
          dataKey="upper"
          stroke="none"
          fill="#AED6F1"
          fillOpacity={0.35}
          name="CI Band"
          legendType="none"
        />
        <Area
          type="monotone"
          dataKey="lower"
          stroke="none"
          fill="#F5F1E8"
          fillOpacity={1}
          name="CI Band"
          legendType="none"
        />

        {/* Forecast line */}
        <Line
          type="monotone"
          dataKey="depth"
          stroke="#1B4F72"
          strokeWidth={2.5}
          dot={{ fill: '#1B4F72', r: 5, strokeWidth: 2, stroke: 'white' }}
          activeDot={{ r: 7, fill: '#C97D3A' }}
          name="Forecast (mbgl)"
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
