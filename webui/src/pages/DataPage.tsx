import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { dataApi } from '../api/endpoints'
import type { Coverage } from '../api/types'
import { Table } from '../components/Table'
import type { Column } from '../components/Table'
import { JobProgress } from '../components/JobProgress'
import { useJob } from '../hooks/useJob'

/** Exclusive update modes; the API rejects force + rebuild_only together. */
type UpdateMode = 'incremental' | 'force' | 'rebuild'

const UPDATE_MODES: UpdateMode[] = ['incremental', 'force', 'rebuild']

const MODE_LABELS: Record<UpdateMode, string> = {
  incremental: 'data.modeIncremental',
  force: 'data.modeForce',
  rebuild: 'data.modeRebuild',
}

export function DataPage() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState<string[]>([])
  const [mode, setMode] = useState<UpdateMode>('incremental')
  const [error, setError] = useState<string | null>(null)
  const job = useJob()

  const { data: coverage, isLoading } = useQuery({
    queryKey: ['data-coverage'],
    queryFn: dataApi.coverage,
  })

  const toggle = (symbol: string) => {
    setSelected((prev) => (prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]))
  }

  const runUpdate = async (symbols: string[]) => {
    // This request really can be refused, and used to be refused in silence:
    // the backend claims each symbol in `_inflight` and answers 409 while a
    // download of it is already running (web/routers/data.py). `job.isActive`
    // does not cover that -- reload the page mid-download and this component's
    // job state is gone, so the buttons come back enabled while the server's
    // claim stands. Unhandled, the rejected promise went to the console and
    // the click looked like it did nothing at all.
    setError(null)
    try {
      const { job_id } = await dataApi.update(symbols, mode === 'force', mode === 'rebuild')
      job.start(job_id)
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  useEffect(() => {
    if (job.state?.status === 'done') {
      void queryClient.invalidateQueries({ queryKey: ['data-coverage'] })
    }
  }, [job.state?.status, queryClient])

  // An empty table is not "all selected": with `coverage` loaded but empty,
  // comparing the two lengths made 0 === 0 true and the header box rendered
  // ticked with nothing to tick.
  const allSelected = (coverage?.length ?? 0) > 0 && selected.length === coverage?.length

  const columns: Column<Coverage>[] = [
    {
      key: 'select',
      header: (
        <input
          type="checkbox"
          checked={allSelected}
          onChange={(e) => setSelected(e.target.checked ? (coverage ?? []).map((c) => c.symbol) : [])}
          title={allSelected ? t('common.deselectAll') : t('common.selectAll')}
        />
      ),
      // The row itself toggles; the box has to swallow its own click or the
      // two handlers would fire in turn and cancel each other out. Keeping
      // the onChange leaves the box keyboard-operable.
      render: (c) => (
        <input
          type="checkbox"
          checked={selected.includes(c.symbol)}
          onChange={() => toggle(c.symbol)}
          onClick={(e) => e.stopPropagation()}
        />
      ),
      width: '32px',
    },
    { key: 'symbol', header: t('data.symbol'), render: (c) => <strong>{c.symbol}</strong> },
    {
      key: 'status',
      header: t('common.status'),
      render: (c) =>
        !c.has_data ? (
          <span className="badge badge-neutral">{t('products.noData')}</span>
        ) : c.stale_keys.length > 0 ? (
          <span className="badge badge-warning">{t('data.stale')}</span>
        ) : (
          <span className="badge badge-success">{t('data.fresh')}</span>
        ),
    },
    { key: 'first_date', header: t('data.firstDate'), render: (c) => c.first_date ?? '—' },
    { key: 'last_date', header: t('data.lastDate'), render: (c) => c.last_date ?? '—' },
    { key: 'n_rows', header: t('data.nRows'), render: (c) => c.n_rows },
    { key: 'last_refresh', header: t('data.lastRefresh'), render: (c) => c.last_refresh ?? '—' },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <h1>{t('data.title')}</h1>
        <p>{t('data.subtitle')}</p>
      </div>

      {error && <div className="hint-banner warning">{t('common.updateFailed')} — {error}</div>}

      <div className="toolbar">
        <span className="toolbar-label">{t('data.mode')}</span>
        {UPDATE_MODES.map((m) => (
          <label key={m} className="checkbox-row" style={{ margin: 0 }}>
            <input
              type="radio"
              name="data-update-mode"
              checked={mode === m}
              onChange={() => setMode(m)}
              disabled={job.isActive}
            />
            {t(MODE_LABELS[m])}
          </label>
        ))}
        <div className="spacer" />
        <button className="btn" onClick={() => void runUpdate(selected)} disabled={selected.length === 0 || job.isActive}>
          {t('data.update')} ({selected.length})
        </button>
        <button className="btn btn-primary" onClick={() => void runUpdate((coverage ?? []).map((c) => c.symbol))} disabled={job.isActive}>
          {t('data.updateAll')}
        </button>
      </div>

      {job.state && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-body">
            <JobProgress
              state={job.state}
              logs={job.logs}
              onCancel={job.cancel}
              streaming={job.streaming}
              lost={job.lost}
            />
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="empty-state">{t('common.loading')}</div>
      ) : (
        <Table
          columns={columns}
          rows={coverage ?? []}
          rowKey={(c) => c.symbol}
          onRowClick={(c) => toggle(c.symbol)}
          rowClassName={(c) => (selected.includes(c.symbol) ? 'row-selectable row-selected' : 'row-selectable')}
        />
      )}
    </div>
  )
}
