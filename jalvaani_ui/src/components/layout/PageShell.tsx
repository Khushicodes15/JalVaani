interface Props {
  title: string
  subtitle?: string
  children: React.ReactNode
  action?: React.ReactNode
}

export function PageShell({ title, subtitle, children, action }: Props) {
  return (
    <div className="flex-1 px-4 sm:px-6 lg:px-8 py-8 max-w-7xl mx-auto w-full animate-fade-in">
      <div className="flex items-start justify-between mb-8 gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-well-900">{title}</h1>
          {subtitle && (
            <p className="text-well-700/60 mt-1 text-sm sm:text-base">{subtitle}</p>
          )}
        </div>
        {action && <div className="flex-shrink-0">{action}</div>}
      </div>
      {children}
    </div>
  )
}
