import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { runsApi } from '../api/endpoints'
import type { RunSummary } from '../api/types'
import { lineSeriesOption } from '../charts/builders'
import { EChart } from '../components/EChart'
import { Table } from '../components/Table'
import type { Column } from '../components/Table'
import { useIsDarkMode } from '../theme/useIsDarkMode'

function formatMetric(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return v.toFixed(4).replace(/\.?0+$/, '') || '0'
  return String(v)
}

export function RunsPage() {
  const { t } = useTranslation()
  const dark = useIsDarkMode()
  const queryClient = useQueryClient()
  const [kindFilter, setKindFilter] = useState<string>('')
  const [selected, setSelected] = useState<string[]>([])
  const [comparing, setComparing] = useState(false)

  const { data: runs, isLoading } = useQuery({
    queryKey: ['runs', kindFilter],
    queryFn: () => runsApi.list(kindFilter || undefined),
  })

  const { data: compareResult } = useQuery({
    queryKey: ['runs-compare', selected],
    queryFn: () => runsApi.compare(selected),
    enabled: comparing && selected.length > 0,
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => runsApi.remove(id),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['runs'] }),
  })

  const toggle = (id: string) => setSelected((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]))

  const allIds = (runs ?? []).map((r) => r.id)
  const allSelected = allIds.length > 0 && allIds.every((id) => selected.includes(id))
  const toggleAll = () => setSelected(allSelected ? [] : allIds)

  const deleteSelected = () => {
    if (selected.length === 0) return
    if (!window.confirm(t('runs.deleteSelectedConfirm', { count: selected.length }))) return
    for (const id of selected) deleteMutation.mutate(id)
    setSelected([])
  }

  const columns: Column<RunSummary>[] = [
    {
      key: 'select',
      header: '',
      render: (r) => <input type="checkbox" checked={selected.includes(r.id)} onChange={() => toggle(r.id)} />,
      width: '32px',
    },
    { key: 'kind', header: t('runs.kind'), render: (r) => <span className="badge badge-neutral">{r.kind}</span> },
    { key: 'strategy', header: t('common.strategy'), render: (r) => r.strategy ?? '—' },
    { key: 'symbols', header: t('common.symbols'), render: (r) => r.symbols.join(', ') },
    { key: 'start', header: t('common.start'), render: (r) => r.start ?? '—' },
    { key: 'end', header: t('common.end'), render: (r) => r.end ?? '—' },
    {
      key: 'status',
      header: t('common.status'),
      render: (r) => (
        <span className={`badge ${r.status === 'done' ? 'badge-success' : r.status === 'error' ? 'badge-danger' : 'badge-neutral'}`}>
          {r.status}
        </span>
      ),
    },
    {
      key: 'metric',
      header: t('backtest.metrics.sharpe_ratio') + ' / ' + t('optimize.bestValue'),
      render: (r) => formatMetric(r.metrics?.[r.kind === 'optimize' ? 'best_value' : 'sharpe_ratio']),
    },
    { key: 'created', header: t('runs.created'), render: (r) => new Date(r.created_at * 1000).toLocaleString() },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <h1>{t('runs.title')}</h1>
        <p>{t('runs.subtitle')}</p>
      </div>

      <div className="toolbar">
        <select value={kindFilter} onChange={(e) => setKindFilter(e.target.value)}>
          <option value="">{t('runs.kind')}: all</option>
          <option value="backtest">backtest</option>
          <option value="optimize">optimize</option>
        </select>
        <div className="spacer" />
        <button className="btn" onClick={toggleAll} disabled={allIds.length === 0}>
          {allSelected ? t('runs.deselectAll') : t('runs.selectAll')}
        </button>
        <button className="btn btn-primary" onClick={() => setComparing(true)} disabled={selected.length === 0}>
          {t('runs.compareSelected')} ({selected.length})
        </button>
        <button className="btn btn-danger" onClick={deleteSelected} disabled={selected.length === 0}>
          {t('runs.deleteSelected')} ({selected.length})
        </button>
      </div>

      {isLoading ? (
        <div className="empty-state">{t('common.loading')}</div>
      ) : (
        <Table columns={columns} rows={runs ?? []} rowKey={(r) => r.id} emptyMessage={t('runs.noRuns')} />
      )}

      {comparing && compareResult && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-header">
            <h3>{t('common.compare')}</h3>
          </div>
          <div className="card-body">
            {Object.keys(compareResult.equity_curves).length > 0 && (
              <EChart
                option={lineSeriesOption(
                  dark,
                  Object.values(compareResult.equity_curves)[0]?.map((p) => p.date) ?? [],
                  Object.entries(compareResult.equity_curves).map(([id, points]) => ({
                    name: compareResult.runs.find((r) => r.id === id)?.strategy ?? id,
                    values: points.map((p) => p.equity),
                  })),
                  { money: true },
                )}
              />
            )}
            <div className="dtable-wrap" style={{ marginTop: 16 }}>
              <table className="dtable">
                <thead>
                  <tr>
                    <th>{t('common.strategy')}</th>
                    <th>{t('backtest.metrics.sharpe_ratio')}</th>
                    <th>{t('backtest.metrics.total_return')}</th>
                    <th>{t('backtest.metrics.max_drawdown')}</th>
                    <th>{t('backtest.metrics.n_trades')}</th>
                  </tr>
                </thead>
                <tbody>
                  {compareResult.runs.map((r) => (
                    <tr key={r.id}>
                      <td>{r.strategy ?? r.id}</td>
                      <td>{formatMetric(r.metrics?.sharpe_ratio)}</td>
                      <td>{formatMetric(r.metrics?.total_return)}</td>
                      <td>{formatMetric(r.metrics?.max_drawdown)}</td>
                      <td>{formatMetric(r.metrics?.n_trades)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
