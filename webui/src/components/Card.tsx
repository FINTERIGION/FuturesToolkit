import type { ReactNode } from 'react'

export function Card({
  title,
  actions,
  children,
}: {
  title?: ReactNode
  actions?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="card">
      {title && (
        <div className="card-header">
          <h3>{title}</h3>
          {actions}
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  )
}
