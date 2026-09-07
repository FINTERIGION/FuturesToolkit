import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { runsApi } from '../../api/endpoints'
import { candlestickOption, lineSeriesOption, positionOption } from '../../charts/builders'
import { Card } from '../../components/Card'
import { EChart } from '../../components/EChart'
import { MetricStats } from '../../components/MetricStats'
import { Table } from '../../components/Table'
import type { Column } from '../../components/Table'
import { Tabs } from '../../components/Tabs'
import { useIsDarkMode } from '../../theme/useIsDarkMode'
import type { RunPrice, TradeLog } from '../../api/types'

export function BacktestResults({ runId }: { runId: string }) {
  const { t } = useTranslation()
  const dark = useIsDarkMode()
  const [tab, setTab] = useState('equity')
  const [priceSymbol, setPriceSymbol] = useState<string | null>(null)
  const [statSymbol, setStatSymbol] = useState<string | null>(null)

  const { data: run } = useQuery({ queryKey: ['run', runId], queryFn: () => runsApi.get(runId) })

  const symbols = run?.symbols_with_price ?? []
  const activeSymbol = priceSymbol ?? symbols[0] ?? null

  const { data: price } = useQuery({
    queryKey: ['run-price', runId, activeSymbol],
    queryFn: () => runsApi.price(runId, activeSymbol as string),
    enabled: !!activeSymbol,
  })

  const equitySeries = useMemo(() => {
    if (!run) return null
    const dates = run.equity_records.map((r) => r.date)
    const equity = run.equity_records.map((r) => r.equity)
    let peak = -Infinity
    const drawdown = equity.map((e) => {
      peak = Math.max(peak, e)
      return peak > 0 ? ((peak - e) / peak) * 100 : 0
    })
    const position: Record<string, number[]> = {}
    for (const rec of run.equity_records) {
      for (const [sym, lots] of Object.entries(rec.position)) {
        ;(position[sym] ??= []).push(lots)
      }
    }
    return { dates, equity, drawdown, position }
  }, [run])

  if (!run) return null

  const tabs = [
    { key: 'equity', label: t('backtest.equityCurve') },
    { key: 'drawdown', label: t('backtest.drawdown') },
    { key: 'position', label: t('backtest.position') },
    { key: 'price', label: t('backtest.priceAndSignals') },
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
    <div className="card">
      <div className="card-body">
        <h3 style={{ marginBottom: 12 }}>{t('backtest.results')}</h3>
        {run.metrics && <MetricStats metrics={run.metrics} />}

        <div style={{ marginTop: 16 }}>
          <Tabs items={tabs} active={tab} onChange={setTab} />

          {tab === 'equity' && equitySeries && (
            <EChart option={lineSeriesOption(dark, equitySeries.dates, [{ name: 'Equity', values: equitySeries.equity }], { area: true, money: true })} />
          )}

          {tab === 'drawdown' && equitySeries && (
            <EChart option={lineSeriesOption(dark, equitySeries.dates, [{ name: 'Drawdown', values: equitySeries.drawdown }], { percent: true, area: true })} />
          )}

          {tab === 'position' && equitySeries && (
            <EChart
              option={positionOption(
                dark,
                equitySeries.dates,
                Object.entries(equitySeries.position).map(([name, values]) => ({ name, values })),
              )}
            />
          )}

          {tab === 'price' && (
            <div>
              <div className="toolbar">
                {symbols.map((sym) => (
                  <button
                    key={sym}
                    className={`btn btn-sm ${activeSymbol === sym ? 'btn-primary' : ''}`}
                    onClick={() => setPriceSymbol(sym)}
                  >
                    {sym}
                  </button>
                ))}
              </div>
              {price && <EChart option={candlestickOption(dark, priceToBars(price), { signals: price.signals })} height={400} />}
            </div>
          )}

          {tab === 'trades' && (
            <Table
              columns={tradeColumns}
              rows={run.trade_logs}
              rowKey={(r) => String(r.trade_id)}
              maxHeight={420}
            />
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
    </div>
  )
}

function priceToBars(price: RunPrice) {
  return price.dates.map((date, i) => ({
    date,
    open: price.open[i],
    high: price.high[i],
    low: price.low[i],
    close: price.close[i],
    settle: price.close[i],
    volume: price.volume[i],
    oi: price.oi[i],
  }))
}
