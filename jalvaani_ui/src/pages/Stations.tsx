import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, ArrowRight, Search } from 'lucide-react'
import { api } from '../api/client'
import { PageShell } from '../components/layout/PageShell'
import { Card } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { FullPageSpinner } from '../components/ui/Spinner'
import { Alert } from '../components/ui/Alert'

const PER_PAGE = 50

export default function Stations() {
  const navigate         = useNavigate()
  const [page, setPage]  = useState(1)
  const [searchQ, setSearchQ]   = useState('')
  const [searchMode, setSearchMode] = useState(false)

  const listQuery = useQuery({
    queryKey: ['stations', page],
    queryFn:  () => api.listStations(page, PER_PAGE),
    enabled:  !searchMode,
    staleTime: 5 * 60 * 1000,
  })

  const searchQuery = useQuery({
    queryKey: ['stations-search', searchQ],
    queryFn:  () => api.searchStations({ q: searchQ, limit: 100 }),
    enabled:  searchMode && searchQ.length >= 2,
    staleTime: 30_000,
  })

  const loading = searchMode ? searchQuery.isFetching : listQuery.isLoading
  const error   = searchMode ? searchQuery.error : listQuery.error
  const stations = searchMode
    ? (searchQuery.data?.stations ?? [])
    : (listQuery.data?.stations  ?? [])

  function handleSearch() {
    setSearchMode(true)
    setPage(1)
  }

  function clearSearch() {
    setSearchQ('')
    setSearchMode(false)
    setPage(1)
  }

  return (
    <PageShell
      title="Station Explorer"
      subtitle={`${(listQuery.data?.total ?? 0).toLocaleString()} CGWB monitoring stations`}
    >
      {/* Search */}
      <Card className="mb-6">
        <label className="block text-xs font-medium text-well-700/60 mb-1.5 uppercase tracking-wider">
          Search stations
        </label>
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-well-400 w-3.5 h-3.5" />
            <input
              value={searchQ}
              onChange={(e) => setSearchQ(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search by city, district or station code — e.g. 'kolkata', 'Jodhpur', 'RAMGARH'"
              className="w-full h-9 pl-8 pr-3 rounded-lg border border-well-200 bg-white text-sm
                         focus:outline-none focus:ring-2 focus:ring-well-500/30 focus:border-well-500"
            />
          </div>
          <Button onClick={handleSearch} size="sm" icon={<Search size={14} />}>Search</Button>
          {searchMode && <Button onClick={clearSearch} variant="secondary" size="sm">Clear</Button>}
        </div>
        <p className="text-xs text-well-700/40 mt-1.5">
          Searches across state, district and station name simultaneously
        </p>
      </Card>

      {loading && <FullPageSpinner label="Loading stations…" />}
      {error && (
        <Alert variant="error">Failed to load stations. Ensure the API server is running.</Alert>
      )}

      {!loading && stations.length > 0 && (
        <>
          {/* Table */}
          <Card padding="sm">
            <div className="overflow-x-auto scrollbar-thin">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-well-100">
                    {['Station', 'State', 'District', 'Lat', 'Lon', ''].map((h) => (
                      <th key={h} className="text-left text-xs uppercase tracking-wider text-well-700/50 font-medium px-3 py-2.5">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {stations.map((s, i) => (
                    <tr
                      key={s.station_name + i}
                      className="border-b border-well-50 hover:bg-well-50 transition-colors"
                    >
                      <td className="px-3 py-2.5 font-medium text-well-900">{s.station_name}</td>
                      <td className="px-3 py-2.5 text-well-700/70">{s.state_name}</td>
                      <td className="px-3 py-2.5 text-well-700/70">{s.district_name}</td>
                      <td className="px-3 py-2.5 text-well-700/50 tabular-nums">{s.latitude.toFixed(3)}</td>
                      <td className="px-3 py-2.5 text-well-700/50 tabular-nums">{s.longitude.toFixed(3)}</td>
                      <td className="px-3 py-2.5">
                        <button
                          onClick={() => navigate(`/forecast?station=${encodeURIComponent(s.station_name)}`)}
                          className="inline-flex items-center gap-1 text-xs text-well-600 hover:text-well-900 font-medium"
                        >
                          Forecast <ArrowRight size={11} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Pagination — list mode only */}
          {!searchMode && listQuery.data && (
            <div className="flex items-center justify-between mt-4 px-1">
              <p className="text-sm text-well-700/60">
                Showing {((page - 1) * PER_PAGE) + 1}–{Math.min(page * PER_PAGE, listQuery.data.total)} of {listQuery.data.total.toLocaleString()}
              </p>
              <div className="flex gap-2">
                <Button
                  variant="secondary" size="sm"
                  disabled={page === 1}
                  onClick={() => setPage(p => p - 1)}
                  icon={<ChevronLeft size={14} />}
                >Prev</Button>
                <Button
                  variant="secondary" size="sm"
                  disabled={page >= listQuery.data.pages}
                  onClick={() => setPage(p => p + 1)}
                >
                  Next <ChevronRight size={14} />
                </Button>
              </div>
            </div>
          )}

          {searchMode && (
            <p className="text-xs text-well-700/50 mt-3 text-center">
              Showing top {stations.length} of {searchQuery.data?.count ?? '?'} results
            </p>
          )}
        </>
      )}

      {!loading && stations.length === 0 && searchMode && (
        <Card className="text-center py-12 text-well-700/40">
          No stations found — try a broader search term.
        </Card>
      )}
    </PageShell>
  )
}
