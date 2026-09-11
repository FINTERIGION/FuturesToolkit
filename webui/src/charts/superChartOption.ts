import type { EChartsOption } from 'echarts'
import type { Bar, RollPoint } from '../api/types'
import { CANDLE } from '../theme/palette'
import { nearestCategory } from './align'
import { baseOption, categoryAxis, chartTokens, valueAxis } from './theme'

/** Panes beyond the always-present price pane, in the order they should
 * stack. `superChartOption` prepends `'price'` itself -- callers only name
 * what they want *added*. Equity/position/drawdown used to live here as run
 * overlays; the equity curve now renders in the backtest panel instead (see
 * panels/ResultTables.tsx), and a run's only contribution to this chart is
 * its buy/sell markers on the price pane. */
export type PaneKey = 'volume'

/** Vertical weight of each pane -- price gets roughly 3-4x a sub-pane's
 * share of the usable height, regardless of how many sub-panes are open. */
const PANE_WEIGHT: Record<PaneKey | 'price', number> = {
  price: 3,
  volume: 0.8,
}

export interface SuperChartInput {
  dark: boolean
  /** The charted product's own full-history bars -- always the base series,
   * whether or not a run is overlaid (see charts/adapters.ts / SuperChart). */
  bars: Bar[]
  /** Sub-panes stacked under the always-present price pane, in display order. */
  panes?: PaneKey[]
  roll?: RollPoint[]
  signals?: { date: string; direction: string }[]
  /** The overlaid run's backtest window, shaded on the price pane. */
  window?: { start: string; end: string } | null
}

/** The single super-chart builder: OHLC + volume, buy/sell markers and the
 * backtest window shading, all sharing one category axis, one linked
 * axisPointer, and one dataZoom. `candlestickOption` below is a thin wrapper
 * kept for the two callers (ProductForm's preview, ProductDrawer.test.tsx)
 * that only ever wanted price+volume. */
export function superChartOption(input: SuperChartInput): EChartsOption {
  const { dark, bars } = input
  const { ink } = chartTokens(dark)
  // Guards against a pane key a returning user's browser still has stored
  // from before a pane was retired (`ft.panes` in localStorage) -- an
  // unrecognized key would otherwise leave `PANE_WEIGHT[p]` undefined and
  // turn the whole layout's height math into NaN.
  const knownPanes = (input.panes ?? []).filter((p): p is PaneKey => p in PANE_WEIGHT)
  const allPanes: (PaneKey | 'price')[] = ['price', ...knownPanes]
  const dates = bars.map((b) => b.date)
  const ohlc = bars.map((b) => [b.open, b.close, b.low, b.high])
  const volumes = bars.map((b) => ({
    value: b.volume,
    itemStyle: { color: b.close >= b.open ? CANDLE.up : CANDLE.down, opacity: 0.5 },
  }))

  const rollBoundaries: string[] = []
  if (input.roll && input.roll.length) {
    let prevContract = input.roll[0].contract
    for (const point of input.roll) {
      if (point.contract !== prevContract) {
        rollBoundaries.push(point.date)
        prevContract = point.contract
      }
    }
  }

  // A signal whose date has no bar in this series is dropped rather than
  // pinned to bar 0 -- the old candlestickOption's `?? 0` fallback, which
  // silently mislocated any signal outside the loaded range.
  const dateIndex = new Map(dates.map((d, i) => [d, i]))
  const knownSignals = (input.signals ?? []).filter((s) => dateIndex.has(s.date))
  const buys = knownSignals.filter((s) => s.direction === 'buy').map((s) => s.date)
  const sells = knownSignals.filter((s) => s.direction === 'sell').map((s) => s.date)
  const priceAt = (d: string) => bars[dateIndex.get(d)!].low

  const windowStart = input.window ? nearestCategory(dates, input.window.start, 'ceil') : null
  const windowEnd = input.window ? nearestCategory(dates, input.window.end, 'floor') : null

  // Fixed pixel gutters on every grid (not `containLabel: true`) so all
  // panes' y-axes land on the same column regardless of label width --
  // percentage-height grids drift apart the moment containLabel resizes one
  // of them differently from the rest.
  const LEFT = 56
  const RIGHT = 20
  const TOP = 2
  const BOTTOM = 11
  const GAP = 2.5
  const totalWeight = allPanes.reduce((s, p) => s + PANE_WEIGHT[p], 0)
  const usable = 100 - TOP - BOTTOM - GAP * (allPanes.length - 1)
  let cursor = TOP
  const rects = allPanes.map((p) => {
    const h = (PANE_WEIGHT[p] / totalWeight) * usable
    const rect = { top: `${cursor}%`, height: `${h}%` }
    cursor += h + GAP
    return rect
  })

  const grid = allPanes.map((_, i) => ({ left: LEFT, right: RIGHT, containLabel: false, ...rects[i] }))
  const xAxisIndexAll = allPanes.map((_, i) => i)

  const xAxis = allPanes.map((_, i) =>
    categoryAxis(dark, dates, { gridIndex: i, axisLabel: { show: i === allPanes.length - 1 } }),
  )

  const yAxis = allPanes.map((pane, i) => {
    if (pane === 'price') {
      return valueAxis(dark, {
        gridIndex: i,
        scale: true,
        axisLabel: { formatter: (v: number) => v.toFixed(2) },
        axisPointer: { label: { formatter: (p: { value: number | string }) => Number(p.value).toFixed(2) } },
      })
    }
    return valueAxis(dark, { gridIndex: i, splitNumber: 2 })
  })

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const series: any[] = []
  allPanes.forEach((pane, i) => {
    if (pane === 'price') {
      series.push({
        type: 'candlestick',
        xAxisIndex: i,
        yAxisIndex: i,
        data: ohlc,
        tooltip: { valueFormatter: (v: unknown) => (typeof v === 'number' ? v.toFixed(2) : String(v)) },
        itemStyle: { color: CANDLE.up, color0: CANDLE.down, borderColor: CANDLE.up, borderColor0: CANDLE.down },
        markPoint:
          buys.length || sells.length
            ? {
                symbolSize: 20,
                data: [
                  ...buys.map((d) => ({
                    name: 'buy', coord: [d, priceAt(d)], symbol: 'triangle', symbolRotate: 0,
                    itemStyle: { color: CANDLE.up }, label: { show: false },
                  })),
                  ...sells.map((d) => ({
                    name: 'sell', coord: [d, priceAt(d)], symbol: 'triangle', symbolRotate: 180,
                    itemStyle: { color: CANDLE.down }, label: { show: false },
                  })),
                ],
              }
            : undefined,
        markLine: rollBoundaries.length
          ? {
              symbol: 'none', silent: true,
              lineStyle: { color: ink.axis, type: 'solid', width: 1, opacity: 0.6 },
              label: { show: false },
              data: rollBoundaries.map((d) => ({ xAxis: d })),
            }
          : undefined,
        markArea:
          windowStart && windowEnd
            ? {
                silent: true,
                itemStyle: { color: dark ? 'rgba(91,139,240,0.10)' : 'rgba(37,99,235,0.07)' },
                data: [[{ xAxis: windowStart }, { xAxis: windowEnd }]],
              }
            : undefined,
      })
    } else if (pane === 'volume') {
      series.push({ type: 'bar', xAxisIndex: i, yAxisIndex: i, data: volumes, barMaxWidth: 6 })
    }
  })

  return {
    ...baseOption(dark),
    legend: { show: false },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    tooltip: { ...baseOption(dark).tooltip, trigger: 'axis', axisPointer: { type: 'cross' } },
    grid,
    xAxis,
    yAxis,
    dataZoom: [
      { type: 'inside', xAxisIndex: xAxisIndexAll },
      { type: 'slider', xAxisIndex: xAxisIndexAll, height: 16, bottom: 4, borderColor: 'transparent' },
    ],
    series,
  }
}
