import { useEffect, useRef, useState } from 'react'
import { Search, X } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../api/client'
import type { Station } from '../../types/api'

interface Props {
  onSelect: (stationName: string) => void
  initialValue?: string
}

export function StationSearch({ onSelect, initialValue = '' }: Props) {
  const [query, setQuery]       = useState(initialValue)
  const [open, setOpen]         = useState(false)
  const [debouncedQ, setDQ]     = useState('')
  const containerRef            = useRef<HTMLDivElement>(null)
  const timerRef                = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Debounce: only fire search 300ms after user stops typing
  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      setDQ(query.trim())
    }, 300)
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [query])

  const { data, isFetching } = useQuery({
    queryKey: ['station-search', debouncedQ],
    queryFn: () => api.searchStations({ q: debouncedQ, limit: 12 }),
    enabled: debouncedQ.length >= 2,
    staleTime: 30_000,
  })

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node))
        setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  function selectStation(s: Station) {
    setQuery(s.station_name)
    setOpen(false)
    onSelect(s.station_name)
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault()
      if (query.trim()) onSelect(query.trim())
      setOpen(false)
    }
    if (e.key === 'Escape') setOpen(false)
  }

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-sm font-medium text-well-800 mb-1.5">Station Name</label>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-well-400 w-4 h-4" />
        <input
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
          onFocus={() => debouncedQ.length >= 2 && setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder="Search by city, district or station code…"
          className="w-full h-10 pl-9 pr-9 rounded-xl border border-well-200 bg-white text-sm
                     focus:outline-none focus:ring-2 focus:ring-well-500/30 focus:border-well-500"
        />
        {query && (
          <button
            onClick={() => { setQuery(''); setOpen(false) }}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-well-400 hover:text-well-700"
          >
            <X size={14} />
          </button>
        )}
        {isFetching && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 border-2 border-well-500 border-t-transparent rounded-full animate-spin" />
        )}
      </div>

      {/* Dropdown */}
      {open && data && data.stations.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-well-200 rounded-xl shadow-well z-50 max-h-64 overflow-y-auto scrollbar-thin">
          {data.stations.map((s) => (
            <button
              key={s.station_name}
              onClick={() => selectStation(s)}
              className="w-full text-left px-4 py-2.5 hover:bg-well-50 transition-colors border-b border-well-50 last:border-0"
            >
              <p className="text-sm font-medium text-well-900">{s.station_name}</p>
              <p className="text-xs text-well-700/50">{s.district_name}, {s.state_name}</p>
            </button>
          ))}
          {data.count > 12 && (
            <p className="text-xs text-center text-well-700/40 py-2">
              Showing 12 of {data.count} — refine your search
            </p>
          )}
        </div>
      )}

      <p className="text-xs text-well-700/50 mt-1.5">
        Type city, district or station code (e.g. "kolkata", "Jodhpur", "RAMGARH1") · Press Enter to look up directly
      </p>
    </div>
  )
}
