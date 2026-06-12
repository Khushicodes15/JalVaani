import { clsx } from 'clsx'
import { AlertCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react'

type AlertVariant = 'info' | 'success' | 'warning' | 'error'

const CONFIG = {
  info:    { icon: Info,          bg: 'bg-well-50',    border: 'border-well-200',    text: 'text-well-700'    },
  success: { icon: CheckCircle,   bg: 'bg-wetland-50', border: 'border-wetland-200', text: 'text-wetland-700' },
  warning: { icon: AlertTriangle, bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700'   },
  error:   { icon: AlertCircle,   bg: 'bg-danger-100', border: 'border-danger-500/30',text: 'text-danger-600' },
}

interface Props {
  variant?: AlertVariant
  title?: string
  children: React.ReactNode
  className?: string
}

export function Alert({ variant = 'info', title, children, className }: Props) {
  const { icon: Icon, bg, border, text } = CONFIG[variant]
  return (
    <div className={clsx(
      'flex gap-3 p-4 rounded-xl border',
      bg, border, text, className
    )}>
      <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        {title && <p className="font-semibold text-sm mb-1">{title}</p>}
        <div className="text-sm opacity-90">{children}</div>
      </div>
    </div>
  )
}
