import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { ApiError, errorMessage } from '../api/client'
import { productsApi, runsApi } from '../api/endpoints'
import type { RunDetail } from '../api/types'
import { superChartOption } from '../charts/superChartOption'
import { EChart } from '../components/EChart'
import { useIsDarkMode } from '../theme/useIsDarkMode'
import { useWorkspace } from '../shell/WorkspaceContext'
import { ChartToolbar } from './ChartToolbar'

/**
 * The main screen: one product's OI-weighted daily candlestick, always
 * loaded from `/products/{code}/bars` (the full history) rather than a
 * run's own price arrays -- the chart's identity is "this product's daily
 * chart" whether or not a run is overlaid, so it must render before any run
 * exists and must not jump when one is added or cleared. A run contributes
 * only its fill signals and its window (shaded via `markArea`) -- its
 * equity curve and trade log render in the backtest panel instead (see
 * panels/ResultTables.tsx).
 */
export function SuperChart({ productName }: { productName?: string }) {
  const { t } = useTranslation()
  const dark = useIsDarkMode()
  const { chartSymbol, setChartSymbol, panes, runId, setRunId } = useWorkspace()

  const { data: barsData, isLoading: barsLoading, error: barsError } = useQuery({
    queryKey: ['product-bars', chartSymbol],
    queryFn: () => productsApi.bars(chartSymbol),
    enabled: Boolean(chartSymbol),
    staleTime: 5 * 60 * 1000,
  })
  const { data: rollData } = useQuery({
    queryKey: ['product-roll', chartSymbol],
    queryFn: () => productsApi.roll(chartSymbol),
    enabled: Boolean(chartSymbol),
    staleTime: 5 * 60 * 1000,
  })

  const { data: run, error: runError } = useQuery<RunDetail>({
    queryKey: ['run', runId],
    queryFn: () => runsApi.get(runId as string),
    enabled: runId !== null,
    retry: false,
  })

  // A run id that has aged out of the store (RUN_RETENTION rolls the index)
  // or was deleted 404s instead of loading -- strip it from the URL rather
  // than leaving a broken overlay chip pointing at nothing.
  useEffect(() => {
    if (runError instanceof ApiError && runError.status === 404) setRunId(null)
  }, [runError, setRunId])

  const symbolHasPrice = Boolean(run?.symbols_with_price.includes(chartSymbol))
  const { data: price } = useQuery({
    queryKey: ['run-price', runId, chartSymbol],
    queryFn: () => runsApi.price(runId as string, chartSymbol),
    enabled: runId !== null && symbolHasPrice,
  })

  if (!chartSymbol) {
    return (
      <div className="chart-area">
        <ChartToolbar />
        <div className="empty-state">{t('workspace.noProducts')}</div>
      </div>
    )
  }

  // Before the loading branch, and checked on its own rather than folded into
  // it: a failed fetch leaves `barsData` undefined with `barsLoading` already
  // back to false, so `barsLoading || !barsData` matches a dead query just as
  // readily as a live one. A product whose history has never been downloaded
  // 404s here, and it sat under "Loading…" forever instead of saying so.
  if (barsError) {
    const isMissing = barsError instanceof ApiError && barsError.status === 404
    return (
      <div className="chart-area">
        <ChartToolbar productName={productName} />
        <div className="empty-state">
          <div>{isMissing ? t('workspace.noDataForSymbol') : errorMessage(barsError)}</div>
          {isMissing && (
            <div style={{ marginTop: 6 }}>{t('workspace.noDataForSymbolHint')}</div>
          )}
        </div>
      </div>
    )
  }

  if (barsLoading || !barsData) {
    return (
      <div className="chart-area">
        <ChartToolbar productName={productName} />
        <div className="empty-state">{t('common.loading')}</div>
      </div>
    )
  }

  const runWindow = run && run.start && run.end ? { start: run.start, end: run.end } : null

  const option = superChartOption({
    dark,
    bars: barsData.bars,
    panes,
    roll: rollData?.roll,
    signals: price?.signals,
    window: runWindow,
  })

  return (
    <div className="chart-area">
      <ChartToolbar productName={productName} />
      {run && !symbolHasPrice && (
        <div className="hint-banner" style={{ margin: '0 12px 8px' }}>
          {t('workspace.runNotOnThisSymbol')}
          {run.symbols[0] && (
            <button className="btn btn-sm" style={{ marginLeft: 8 }} onClick={() => setChartSymbol(run.symbols[0])}>
              {t('workspace.chartThisRunsSymbol', { symbol: run.symbols[0] })}
            </button>
          )}
        </div>
      )}
      <div className="super-chart-canvas">
        <EChart option={option} preserveDataZoom />
      </div>
    </div>
  )
}
