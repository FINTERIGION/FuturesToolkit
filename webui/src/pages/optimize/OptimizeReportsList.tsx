import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { errorMessage } from '../../api/client'
import { optimizeApi } from '../../api/endpoints'
import type { OptimizeReportSummary } from '../../api/types'
import { Drawer } from '../../components/Drawer'
import { MetricStats } from '../../components/MetricStats'
import { Table } from '../../components/Table'
import type { Column } from '../../components/Table'

export function OptimizeReportsList() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: reports } = useQuery({ queryKey: ['optimize-reports'], queryFn: optimizeApi.reports })

  const [viewName, setViewName] = useState<string | null>(null)
  const { data: viewReport, isLoading: viewLoading } = useQuery({
    queryKey: ['optimize-report', viewName],
    queryFn: () => optimizeApi.report(viewName as string),
    enabled: viewName !== null,
  })

  const holdoutMutation = useMutation({
    mutationFn: (name: string) => optimizeApi.holdout(name),
    onSuccess: (_data, name) => {
      void queryClient.invalidateQueries({ queryKey: ['optimize-reports'] })
      void queryClient.invalidateQueries({ queryKey: ['optimize-report', name] })
      setViewName(name)
    },
    onError: (err) => window.alert(errorMessage(err)),
  })

  const runHoldout = (report: OptimizeReportSummary) => {
    if (report.holdout_evaluated) {
      if (!window.confirm(t('optimize.holdoutConfirm'))) return
      holdoutMutation.mutate(report.name)
      return
    }
    holdoutMutation.mutate(report.name)
  }

  const sendToBacktest = async (name: string) => {
    const full = await optimizeApi.report(name)
    navigate('/backtest', {
      state: {
        strategy: full.strategy_key,
        symbols: full.symbols,
        start: full.start,
        end: full.end,
        params: full.best_params,
      },
    })
  }

  const columns: Column<OptimizeReportSummary>[] = [
    { key: 'strategy', header: t('common.strategy'), render: (r) => r.strategy },
    { key: 'symbols', header: t('common.symbols'), render: (r) => r.symbols.join(', ') },
    { key: 'timestamp', header: t('runs.created'), render: (r) => r.timestamp },
    { key: 'best_value', header: t('optimize.bestValue'), render: (r) => r.best_value.toFixed(4) },
    {
      key: 'holdout',
      header: t('optimize.runHoldout'),
      render: (r) => (
        <span className={`badge ${r.holdout_evaluated ? 'badge-warning' : 'badge-neutral'}`}>
          {r.holdout_evaluated ? t('common.yes') : t('common.no')}
        </span>
      ),
    },
    {
      key: 'actions',
      header: t('common.actions'),
      render: (r) => (
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="btn btn-sm" onClick={() => setViewName(r.name)}>
            {t('common.view')}
          </button>
          <button className="btn btn-sm" onClick={() => runHoldout(r)} disabled={holdoutMutation.isPending}>
            {t('optimize.runHoldout')}
          </button>
          <button className="btn btn-sm btn-primary" onClick={() => void sendToBacktest(r.name)}>
            {t('optimize.sendToBacktest')}
          </button>
        </div>
      ),
    },
  ]

  return (
    <>
      <Table columns={columns} rows={reports ?? []} rowKey={(r) => r.name} />
      <Drawer open={viewName !== null} onClose={() => setViewName(null)} title={viewName ?? ''}>
        {viewLoading && <p>{t('common.loading')}</p>}
        {viewReport && (
          <>
            <div className="stat-grid" style={{ marginBottom: 16 }}>
              <div className="stat-tile">
                <div className="label">{t('optimize.bestValue')}</div>
                <div className="value">{viewReport.best_value.toFixed(4)}</div>
              </div>
              <div className="stat-tile">
                <div className="label">{t('optimize.pbo')}</div>
                <div className="value">{viewReport.diagnostics.pbo.pbo?.toFixed(3) ?? t('common.na')}</div>
              </div>
              <div className="stat-tile">
                <div className="label">{t('optimize.dsr')}</div>
                <div className="value">{viewReport.diagnostics.dsr.dsr?.toFixed(3) ?? t('common.na')}</div>
              </div>
            </div>
            <h3 style={{ marginBottom: 10 }}>{t('optimize.holdoutResults')}</h3>
            {viewReport.holdout_evaluated && viewReport.holdout_metrics ? (
              <>
                <div className="stat-grid" style={{ marginBottom: 12 }}>
                  <div className="stat-tile">
                    <div className="label">{t('optimize.holdoutScore')}</div>
                    <div className="value">{viewReport.holdout_score?.toFixed(4) ?? t('common.na')}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="label">{t('optimize.holdoutRuns')}</div>
                    <div className="value">{viewReport.holdout_runs}</div>
                  </div>
                </div>
                <MetricStats metrics={viewReport.holdout_metrics} />
              </>
            ) : (
              <p className="hint-banner">{t('optimize.holdoutNotEvaluated')}</p>
            )}
            <h3 style={{ margin: '16px 0 10px' }}>{t('optimize.bestParams')}</h3>
            <pre style={{ fontSize: 12, background: 'var(--surface-2)', padding: 10, borderRadius: 6, overflowX: 'auto' }}>
              {JSON.stringify(viewReport.best_params, null, 2)}
            </pre>
          </>
        )}
      </Drawer>
    </>
  )
}
