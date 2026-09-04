import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { dataApi } from '../api/endpoints'
import type { Coverage } from '../api/types'
import { Table } from '../components/Table'
import type { Column } from '../components/Table'
import { JobProgress } from '../components/JobProgress'
import { useJob } from '../hooks/useJob'

export function DataPage() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState<string[]>([])
  const [force, setForce] = useState(false)
  const [rebuildOnly, setRebuildOnly] = useState(false)
  const job = useJob()

  const { data: coverage, isLoading } = useQuery({
    queryKey: ['data-coverage'],
    queryFn: dataApi.coverage,
  })

  const toggle = (symbol: string) => {
    setSelected((prev) => (prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]))
  }

  const runUpdate = async (symbols: string[]) => {
    const { job_id } = await dataApi.update(symbols, force, rebuildOnly)
    job.start(job_id)
  }

  useEffect(() => {
    if (job.state?.status === 'done') {
      void queryClient.invalidateQueries({ queryKey: ['data-coverage'] })
    }
  }, [job.state?.status, queryClient])

  const columns: Column<Coverage>[] = [
    {
      key: 'select',
      header: <input type="checkbox" checked={selected.length === (coverage?.length ?? -1)} onChange={(e) => setSelected(e.target.checked ? (coverage ?? []).map((c) => c.symbol) : [])} />,
      render: (c) => <input type="checkbox" checked={selected.includes(c.symbol)} onChange={() => toggle(c.symbol)} />,
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

      <div className="toolbar">
        <label className="checkbox-row" style={{ margin: 0 }}>
          <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
          {t('data.force')}
        </label>
        <label className="checkbox-row" style={{ margin: 0 }}>
          <input type="checkbox" checked={rebuildOnly} onChange={(e) => setRebuildOnly(e.target.checked)} />
          {t('data.rebuildOnly')}
        </label>
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
            <JobProgress state={job.state} logs={job.logs} onCancel={job.cancel} streaming={job.streaming} />
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="empty-state">{t('common.loading')}</div>
      ) : (
        <Table columns={columns} rows={coverage ?? []} rowKey={(c) => c.symbol} />
      )}
    </div>
  )
}
