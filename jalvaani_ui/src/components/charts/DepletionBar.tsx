import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts'

interface Props {
  data: Record<string, number>
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const pct = Math.round(payload[0].value * 100)
  return (
    <div className="bg-white border border-well-200 rounded-xl shadow-well px-4 py-3 text-sm">
      <p className="font-semibold text-well-900">{label}</p>
      <p className="text-danger-600 font-bold mt-1">{pct}% depleting stations</p>
    </div>
  )
}

function barColor(value: number): string {
  if (value >= 0.7) return '#C0392B'
  if (value >= 0.5) return '#D97706'
  if (value >= 0.35) return '#F59E0B'
  return '#1E7A4B'
}

export function DepletionBar({ data }: Props) {
  const chartData = Object.entries(data)
    .map(([state, pct]) => ({ state, pct }))
    .sort((a, b) => b.pct - a.pct)

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart
        layout="vertical"
        data={chartData}
        margin={{ top: 0, right: 24, left: 8, bottom: 0 }}
        barSize={16}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#D6EAF8" />
        <XAxis
          type="number"
          domain={[0, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          tick={{ fontSize: 11, fill: '#1B4F72', opacity: 0.7 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="state"
          tick={{ fontSize: 12, fill: '#1B4F72' }}
          axisLine={false}
          tickLine={false}
          width={110}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: '#EBF5FB' }} />
        <Bar dataKey="pct" radius={[0, 6, 6, 0]}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={barColor(entry.pct)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
