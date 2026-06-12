import { clsx } from 'clsx'

type Variant = 'safe' | 'medium' | 'high' | 'reliable' | 'low_confidence' |
               'depleting' | 'stable' | 'recovering' | 'unknown' | 'neutral'

const STYLES: Record<Variant, string> = {
  safe:          'bg-wetland-100 text-wetland-700 border-wetland-400/30',
  medium:        'bg-amber-100 text-amber-600 border-amber-400/40',
  high:          'bg-danger-100 text-danger-600 border-danger-500/30',
  reliable:      'bg-wetland-100 text-wetland-700 border-wetland-400/30',
  low_confidence:'bg-amber-100 text-amber-600 border-amber-400/40',
  depleting:     'bg-danger-100 text-danger-600 border-danger-500/30',
  stable:        'bg-well-100 text-well-700 border-well-300/40',
  recovering:    'bg-wetland-100 text-wetland-700 border-wetland-400/30',
  unknown:       'bg-gray-100 text-gray-500 border-gray-300/40',
  neutral:       'bg-well-100 text-well-700 border-well-300/40',
}

interface Props {
  variant: Variant
  children: React.ReactNode
  className?: string
  dot?: boolean
}

export function Badge({ variant, children, className, dot }: Props) {
  return (
    <span className={clsx(
      'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border',
      STYLES[variant],
      className
    )}>
      {dot && (
        <span className={clsx('w-1.5 h-1.5 rounded-full flex-shrink-0', {
          'bg-wetland-500': variant === 'safe' || variant === 'reliable' || variant === 'recovering',
          'bg-amber-500':   variant === 'medium' || variant === 'low_confidence',
          'bg-danger-500':  variant === 'high' || variant === 'depleting',
          'bg-well-500':    variant === 'stable' || variant === 'neutral',
          'bg-gray-400':    variant === 'unknown',
        })} />
      )}
      {children}
    </span>
  )
}
