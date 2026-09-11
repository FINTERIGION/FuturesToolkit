import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { RunDetail, TradeLog } from '../api/types'
import { lineSeriesOption } from '../charts/builders'
import { Card } from '../components/Card'
import { EChart } from '../components/EChart'
import { MetricStats } from '../components/MetricStats'
import { Table } from '../components/Table'
import type { Column } from '../components/Table'
import { Tabs } from '../components/Tabs'
import { useIsDarkMode } from '../theme/useIsDarkMode'

/**
 * The tabular half of a backtest result -- metric tiles, trade log, and the
 * by-symbol / by-exit-reason breakdowns, plus the equity curve. Price &
 * signals stay on the main super chart (see chart/SuperChart.tsx), overlaid
 * on whichever product is charted -- but the equity curve is the run's own
 * portfolio-level series, not tied to any one charted symbol, so it draws
 * here instead.
 */
export function ResultTables({ run }: { run: RunDetail }) {
  const { t } = useTranslation()
  const dark = useIsDarkMode()
  const [tab, setTab] = useState('trades')
  const [statSymbol, setStatSymbol] = useState<string | null>(null)

  const equityOption = useMemo(
    () =>
      lineSeriesOption(
        dark,
        run.equity_records.map((r) => r.date),
        [{ name: t('backtest.equityCurve'), values: run.equity_records.map((r) => r.equity) }],
        { money: true, area: true },
      ),
    [dark, run.equity_records, t],
  )

  const tabs = [
    { key: 'trades', label: t('backtest.tradeLog') },
    { key: 'symbol', label: t('backtest.bySymbol') },
    { key: 'exit', label: t('backtest.byExitReason') },
  ]

  const tradeColumns: Column<TradeLog>[] = [
    { key: 'symbol', header: t('common.symbols'), render: (r) => r.symbol },
    { key: 'direction', header: t('backtest.direction'), render: (r) => r.direction },
    { key: 'open_date', header: t('common.start'), render: (r) => r.open_date },
    { key: 'close_date', header: t('common.end'), render: (r) => r.close_date ?? '—' },
    { key: 'size', header: 'Size', render: (r) => r.size },
    { key: 'net_pnl', header: 'Net PnL', render: (r) => r.net_pnl.toFixed(2) },
    { key: 'exit_reason', header: 'Exit', render: (r) => r.exit_reason },
  ]

  const bySymbol = (run.metrics?.['by_symbol'] as Record<string, Record<string, number>>) ?? {}
  const byExit = (run.metrics?.['by_exit_reason'] as Record<string, Record<string, number>>) ?? {}
  const bySymbolKeys = Object.keys(bySymbol)
  const activeStatSymbol = statSymbol ?? bySymbolKeys[0] ?? null

  return (
    <div>
      {/* Above the metrics, not below: everything under this banner
          describes a window that ended early, and a run that never recorded
          a bar at all (cash <= 0) has no metrics under it whatsoever -- an
          unexplained grid of "n/a" is exactly what this replaces. */}
      {Boolean(run.metrics?.blown_up) && (
        <div className="hint-banner danger">
          <strong>{t('backtest.blownUp')}</strong> — {t('backtest.blownUpHint')}
        </div>
      )}
      {run.equity_records.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <Card title={t('backtest.equityCurve')}>
            <EChart option={equityOption} height={220} />
          </Card>
        </div>
      )}
      {run.metrics && <MetricStats metrics={run.metrics} />}

      <div style={{ marginTop: 16 }}>
        <Tabs items={tabs} active={tab} onChange={setTab} />

        {tab === 'trades' && (
          <Table columns={tradeColumns} rows={run.trade_logs} rowKey={(r) => String(r.trade_id)} maxHeight={420} />
        )}

        {tab === 'symbol' && (
          <div>
            <div className="toolbar">
              {bySymbolKeys.map((sym) => (
                <button
                  key={sym}
                  className={`btn btn-sm ${activeStatSymbol === sym ? 'btn-primary' : ''}`}
                  onClick={() => setStatSymbol(sym)}
                >
                  {sym}
                </button>
              ))}
            </div>
            {activeStatSymbol && bySymbol[activeStatSymbol] && (
              <Card title={activeStatSymbol}>
                <MetricStats metrics={bySymbol[activeStatSymbol]} keys={Object.keys(bySymbol[activeStatSymbol])} />
              </Card>
            )}
          </div>
        )}

        {tab === 'exit' && (
          <div className="grid-2">
            {Object.entries(byExit).map(([reason, stats]) => (
              <Card title={reason} key={reason}>
                <MetricStats metrics={stats} keys={Object.keys(stats)} />
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
