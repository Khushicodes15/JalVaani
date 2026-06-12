import { clsx } from 'clsx'

interface Props {
  size?: 'sm' | 'md' | 'lg'
  className?: string
  label?: string
}

const SIZES = { sm: 'w-4 h-4', md: 'w-8 h-8', lg: 'w-12 h-12' }

export function Spinner({ size = 'md', className, label }: Props) {
  return (
    <div className={clsx('flex flex-col items-center gap-3', className)}>
      <div className={clsx(
        'border-3 border-well-100 border-t-well-600 rounded-full animate-spin',
        SIZES[size]
      )} style={{ borderWidth: size === 'sm' ? 2 : 3 }} />
      {label && <p className="text-sm text-well-700/60">{label}</p>}
    </div>
  )
}

export function FullPageSpinner({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex-1 flex items-center justify-center min-h-64">
      <Spinner size="lg" label={label} />
    </div>
  )
}
