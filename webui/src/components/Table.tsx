import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'

export interface Column<T> {
  key: string
  header: ReactNode
  render: (row: T) => ReactNode
  width?: string
}

export function Table<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  rowClassName,
  emptyMessage,
  maxHeight,
}: {
  columns: Column<T>[]
  rows: T[]
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  rowClassName?: (row: T) => string | undefined
  emptyMessage?: string
  maxHeight?: number | string
}) {
  const { t } = useTranslation()
  if (rows.length === 0) {
    return <div className="empty-state">{emptyMessage ?? t('common.none')}</div>
  }
  return (
    <div className="dtable-wrap" style={maxHeight ? { maxHeight, overflowY: 'auto' } : undefined}>
      <table className="dtable">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.key} style={{ width: col.width }}>
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              className={rowClassName?.(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              style={onRowClick ? { cursor: 'pointer' } : undefined}
            >
              {columns.map((col) => (
                <td key={col.key}>{col.render(row)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
