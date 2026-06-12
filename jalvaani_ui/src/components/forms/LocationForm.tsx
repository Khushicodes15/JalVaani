import { useState } from 'react'
import { MapPin } from 'lucide-react'
import { Button } from '../ui/Button'
import type { LocationInput } from '../../types/api'

// Quick-pick locations for common Indian cities
const PRESETS = [
  { label: 'Delhi',     lat: 28.6,  lon: 77.2,  date: '2020-08-15' },
  { label: 'Mumbai',    lat: 19.07, lon: 72.87, date: '2020-07-20' },
  { label: 'Chennai',   lat: 13.08, lon: 80.27, date: '2020-09-01' },
  { label: 'Kolkata',   lat: 22.57, lon: 88.36, date: '2020-08-10' },
  { label: 'Jaipur',    lat: 26.91, lon: 75.79, date: '2020-05-15' },
  { label: 'Bengaluru', lat: 12.97, lon: 77.59, date: '2020-07-01' },
]

const today = new Date().toISOString().split('T')[0]

interface Props {
  onSubmit: (loc: LocationInput) => void
  loading?: boolean
  submitLabel?: string
}

export function LocationForm({ onSubmit, loading, submitLabel = 'Run Analysis' }: Props) {
  const [form, setForm] = useState<LocationInput>({
    latitude:  28.6,
    longitude: 77.2,
    date:      '2020-08-15',
  })
  const [errors, setErrors] = useState<Partial<Record<keyof LocationInput, string>>>({})
  const [selectedPreset, setSelectedPreset] = useState<string>('Delhi')

  function validate(): boolean {
    const e: typeof errors = {}
    if (form.latitude < 6 || form.latitude > 38)
      e.latitude = 'Must be 6.0–38.0 (India)'
    if (form.longitude < 68 || form.longitude > 98)
      e.longitude = 'Must be 68.0–98.0 (India)'
    if (!/^\d{4}-\d{2}-\d{2}$/.test(form.date))
      e.date = 'Use YYYY-MM-DD'
    const yr = new Date(form.date).getFullYear()
    if (yr < 2000 || yr > 2030) e.date = 'Year must be 2000–2030'
    setErrors(e)
    return Object.keys(e).length === 0
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (validate()) onSubmit(form)
  }

  function applyPreset(p: typeof PRESETS[0]) {
    setForm({ latitude: p.lat, longitude: p.lon, date: p.date })
    setSelectedPreset(p.label)
    setErrors({})
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* Presets */}
      <div>
        <p className="text-xs font-medium text-well-700/50 mb-2 uppercase tracking-wider">
          Quick locations
        </p>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => {
            const active = selectedPreset === p.label
            return (
              <button
                key={p.label}
                type="button"
                onClick={() => applyPreset(p)}
                className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium
                           border transition-colors
                           ${active
                             ? 'bg-well-700 text-white border-well-700 shadow-sm'
                             : 'bg-well-50 text-well-700 border-well-200 hover:bg-well-100'}`}
              >
                <MapPin size={11} />
                {p.label}
              </button>
            )
          })}
        </div>
      </div>

      {/* Inputs — lat/lon on row 1, date full-width on row 2 */}
      <div className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[
            { key: 'latitude',  label: 'Latitude',  placeholder: '28.6', step: '0.001', min: '6',  max: '38' },
            { key: 'longitude', label: 'Longitude', placeholder: '77.2', step: '0.001', min: '68', max: '98' },
          ].map(({ key, label, placeholder, step, min, max }) => (
            <div key={key}>
              <label className="block text-sm font-medium text-well-800 mb-1.5">{label}</label>
              <input
                type="number"
                step={step}
                min={min}
                max={max}
                value={form[key as keyof LocationInput]}
                onChange={(e) => {
                  setForm({ ...form, [key]: parseFloat(e.target.value) })
                  setSelectedPreset('')
                }}
                placeholder={placeholder}
                className="w-full h-10 px-3 rounded-xl border border-well-200 bg-white text-sm
                           focus:outline-none focus:ring-2 focus:ring-well-500/30 focus:border-well-500
                           transition-colors"
              />
              {errors[key as keyof LocationInput] && (
                <p className="text-xs text-danger-600 mt-1">{errors[key as keyof LocationInput]}</p>
              )}
            </div>
          ))}
        </div>

        <div>
          <label className="block text-sm font-medium text-well-800 mb-1.5">Date</label>
          <input
            type="date"
            value={form.date}
            min="2000-01-01"
            max={today}
            onChange={(e) => {
              setForm({ ...form, date: e.target.value })
              setSelectedPreset('')
            }}
            className="w-full h-10 px-3 rounded-xl border border-well-200 bg-white text-sm
                       focus:outline-none focus:ring-2 focus:ring-well-500/30 focus:border-well-500
                       transition-colors"
          />
          {errors.date && <p className="text-xs text-danger-600 mt-1">{errors.date}</p>}
        </div>
      </div>

      <Button type="submit" loading={loading} size="lg" className="w-full sm:w-auto">
        {submitLabel}
      </Button>
    </form>
  )
}
