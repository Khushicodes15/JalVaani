import { Droplets, Menu, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { clsx } from 'clsx'

const NAV_ITEMS = [
  { to: '/',            label: 'Dashboard' },
  { to: '/depth',       label: 'Depth' },
  { to: '/contamination', label: 'Contamination' },
  { to: '/forecast',    label: 'Forecast' },
  { to: '/report',      label: 'Full Report' },
  { to: '/stations',    label: 'Stations' },
]

export function Navbar() {
  const [open, setOpen] = useState(false)

  return (
    <nav className="bg-well-900 text-white shadow-lg relative z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center h-16 gap-6">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2.5 flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-water-gradient flex items-center justify-center">
              <Droplets className="w-4.5 h-4.5 text-white" size={18} />
            </div>
            <div>
              <span className="font-bold text-lg leading-none tracking-tight">JalVaani</span>
              <span className="block text-[10px] text-well-300 leading-none font-normal tracking-widest uppercase">
                AI
              </span>
            </div>
          </NavLink>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-1 flex-1">
            {NAV_ITEMS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => clsx(
                  'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-well-700 text-white'
                    : 'text-well-300 hover:text-white hover:bg-well-800'
                )}
              >
                {label}
              </NavLink>
            ))}
          </div>

          {/* Tagline — desktop only */}
          <p className="hidden lg:block text-well-400 text-xs ml-auto flex-shrink-0">
            ~903k CGWB readings · 16,693 stations
          </p>

          {/* Mobile menu button */}
          <button
            className="md:hidden ml-auto p-2 rounded-lg hover:bg-well-800 transition-colors"
            onClick={() => setOpen(!open)}
            aria-label="Toggle menu"
          >
            {open ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden border-t border-well-800 bg-well-900 px-4 pb-4 pt-2 space-y-1">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => setOpen(false)}
              className={({ isActive }) => clsx(
                'block px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-well-700 text-white'
                  : 'text-well-300 hover:text-white hover:bg-well-800'
              )}
            >
              {label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  )
}
