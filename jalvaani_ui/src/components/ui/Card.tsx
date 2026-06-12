import { clsx } from 'clsx'

interface CardProps {
  children: React.ReactNode
  className?: string
  padding?: 'sm' | 'md' | 'lg'
  hover?: boolean
}

export function Card({ children, className, padding = 'md', hover }: CardProps) {
  return (
    <div className={clsx(
      'bg-white rounded-2xl border border-well-100 shadow-card',
      {
        'p-4':   padding === 'sm',
        'p-6':   padding === 'md',
        'p-8':   padding === 'lg',
        'transition-shadow hover:shadow-well cursor-pointer': hover,
      },
      className
    )}>
      {children}
    </div>
  )
}

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  icon?: React.ReactNode
  accent?: 'blue' | 'earth' | 'green' | 'red'
}

const ACCENT_STYLES = {
  blue:  'bg-well-700',
  earth: 'bg-soil-600',
  green: 'bg-wetland-600',
  red:   'bg-danger-600',
}

export function StatCard({ label, value, sub, icon, accent = 'blue' }: StatCardProps) {
  return (
    <Card>
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-sm text-well-700/60 font-medium truncate">{label}</p>
          <p className="stat-value mt-1">{value}</p>
          {sub && <p className="text-xs text-well-700/50 mt-1">{sub}</p>}
        </div>
        {icon && (
          <div className={clsx(
            'w-11 h-11 rounded-xl flex items-center justify-center text-white flex-shrink-0 ml-3',
            ACCENT_STYLES[accent]
          )}>
            {icon}
          </div>
        )}
      </div>
    </Card>
  )
}
