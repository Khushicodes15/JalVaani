import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Navbar } from './components/layout/Navbar'
import { FullPageSpinner } from './components/ui/Spinner'

// Code-split: each page is its own chunk, loaded only when visited
const Dashboard    = lazy(() => import('./pages/Dashboard'))
const DepthPredictor = lazy(() => import('./pages/DepthPredictor'))
const Contamination = lazy(() => import('./pages/Contamination'))
const Forecast     = lazy(() => import('./pages/Forecast'))
const FullReport   = lazy(() => import('./pages/FullReport'))
const Stations     = lazy(() => import('./pages/Stations'))

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-sand">
        <Navbar />
        <Suspense fallback={<FullPageSpinner />}>
          <Routes>
            <Route path="/"              element={<Dashboard />} />
            <Route path="/depth"         element={<DepthPredictor />} />
            <Route path="/contamination" element={<Contamination />} />
            <Route path="/forecast"      element={<Forecast />} />
            <Route path="/report"        element={<FullReport />} />
            <Route path="/stations"      element={<Stations />} />
            {/* Catch-all */}
            <Route path="*" element={
              <div className="flex-1 flex items-center justify-center text-well-700/40">
                <div className="text-center">
                  <p className="text-6xl mb-4">404</p>
                  <p>Page not found.</p>
                </div>
              </div>
            } />
          </Routes>
        </Suspense>
      </div>
    </BrowserRouter>
  )
}
