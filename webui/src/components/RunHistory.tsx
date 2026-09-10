import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { runsApi } from '../api/endpoints'
import type { RunSummary } from '../api/types'
import { Table } from './Table'
import type { Column } from './Table'

function formatMetric(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return v.toFixed(4).replace(/\.?0+$/, '') || '0'
  return String(v)
}

/** The backtest run index, rendered under the page that produces it --
 * there is no separate Runs tab, so neither a kind filter nor a kind column
 * is needed here.
 *
 * Optimize runs are deliberately not listed: their durable record is the
 * report file under `results/optuna/`, which the Optimize page's Past Reports
 * list reads straight off disk. The run index could not stand in for it --
 * it is a rolling window (`RUN_RETENTION`) that drops old rows, it misses
 * studies run from the CLI entirely, and an optimize row carries no equity
 * curve, so the results panel would have nothing to draw.
 *
 * Rows behave like Past Reports' rows -- the row itself is the way in, and the
 * only per-row button is Delete -- but the run they open is shown in the
 * page's own results panel rather than a drawer: this detail view is a full
 * chart panel, and an overlay wide enough to hold it stops reading as one. */
export function RunHistory({
  onOpen,
  onDeleted,
  openRunId,
}: {
  onOpen?: (runId: string) => void
  onDeleted?: (runId: string) => void
  openRunId?: string | null
}) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()

  const { data: runs, isLoading } = useQuery({
    queryKey: ['runs', 'backtest'],
    queryFn: () => runsApi.list('backtest'),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => runsApi.remove(id),
    onSuccess: (_data, id) => {
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      // The results panel may be reading a run that no longer exists; leaving
      // it up would sit on a stale copy of charts nothing can reload.
      onDeleted?.(id)
    },
    onError: (err) => window.alert(errorMessage(err)),
  })

  const deleteRun = (r: RunSummary) => {
    if (!window.confirm(t('runs.deleteRunConfirm', { strategy: r.strategy ?? r.id }))) return
    deleteMutation.mutate(r.id)
  }

  // Only a finished run has an artifact to open; a queued/errored one would
  // load into empty charts, so its row is inert.
  const openable = (r: RunSummary) => Boolean(onOpen) && r.status === 'done'

  const columns: Column<RunSummary>[] = [
    { key: 'strategy', header: t('common.strategy'), render: (r) => r.strategy ?? '—' },
    { key: 'symbols', header: t('common.symbols'), render: (r) => r.symbols.join(', ') },
    { key: 'start', header: t('common.start'), render: (r) => r.start ?? '—' },
    { key: 'end', header: t('common.end'), render: (r) => r.end ?? '—' },
    {
      key: 'status',
      header: t('common.status'),
      // A blown-up run finished cleanly as a job, so its row status is `done`.
      // Showing that in green here would put a run with no usable numbers next
      // to real ones under the same badge, which is the whole reason the flag
      // is carried this far.
      render: (r) =>
        r.status === 'done' && r.metrics?.blown_up ? (
          <span className="badge badge-danger">{t('backtest.blownUp')}</span>
        ) : (
          <span className={`badge ${r.status === 'done' ? 'badge-success' : r.status === 'error' ? 'badge-danger' : 'badge-neutral'}`}>
            {r.status}
          </span>
        ),
    },
    {
      key: 'metric',
      header: t('backtest.metrics.sharpe_ratio'),
      render: (r) => formatMetric(r.metrics?.sharpe_ratio),
    },
    { key: 'created', header: t('runs.created'), render: (r) => new Date(r.created_at * 1000).toLocaleString() },
    {
      key: 'actions',
      header: t('common.actions'),
      // The button sits inside the row's own click target, so it has to
      // swallow its own click -- without that, confirming a delete would open
      // the deleted run in the results panel on the way out.
      render: (r) => (
        <button
          className="btn btn-sm btn-danger"
          onClick={(e) => {
            e.stopPropagation()
            deleteRun(r)
          }}
          disabled={deleteMutation.isPending}
        >
          {t('common.delete')}
        </button>
      ),
    },
  ]

  if (isLoading) return <div className="empty-state">{t('common.loading')}</div>

  return (
    // The row is the way into the results panel -- there is no View button,
    // so `row-selectable` is what stops a click that lands beside the Delete
    // button from selecting the cell text instead.
    <Table
      columns={columns}
      rows={runs ?? []}
      rowKey={(r) => r.id}
      emptyMessage={t('runs.noRuns')}
      onRowClick={(r) => {
        if (openable(r)) onOpen?.(r.id)
      }}
      rowClassName={(r) =>
        [openRunId === r.id ? 'row-open' : '', openable(r) ? 'row-selectable' : 'row-static'].filter(Boolean).join(' ')
      }
      maxHeight={360}
    />
  )
}
