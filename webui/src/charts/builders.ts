import type { EChartsOption } from 'echarts'
import type { Bar, RollPoint } from '../api/types'
import { superChartOption } from './superChartOption'
import { baseOption, categoryAxis, chartTokens, valueAxis } from './theme'

/** One or more named series against a shared date axis -- an equity curve,
 * a drawdown line, or a multi-run comparison. A single series carries no
 * legend (the card title already names it); two or more always do. */
export function lineSeriesOption(
  dark: boolean,
  dates: string[],
  series: { name: string; values: (number | null)[] }[],
  opts: { percent?: boolean; area?: boolean; money?: boolean } = {},
): EChartsOption {
  const { categorical } = chartTokens(dark)
  const showLegend = series.length > 1
  return {
    ...baseOption(dark),
    legend: showLegend ? { ...baseOption(dark).legend, data: series.map((s) => s.name) } : { show: false },
    grid: { ...(baseOption(dark).grid as object), top: showLegend ? 36 : 20 },
    tooltip: {
      ...baseOption(dark).tooltip,
      trigger: 'axis',
      ...(opts.money ? { valueFormatter: (v: unknown) => (typeof v === 'number' ? v.toFixed(2) : String(v)) } : {}),
    },
    xAxis: categoryAxis(dark, dates),
    yAxis: valueAxis(
      dark,
      opts.percent
        ? { axisLabel: { formatter: '{value}%' } }
        : opts.money
          ? { axisLabel: { formatter: (v: number) => v.toFixed(2) } }
          : {},
    ),
    dataZoom: [
      { type: 'inside' },
      { type: 'slider', height: 18, bottom: 4, borderColor: 'transparent' },
    ],
    series: series.map((s, i) => ({
      name: s.name,
      type: 'line',
      data: s.values,
      showSymbol: false,
      lineStyle: { width: 2, color: categorical[i % categorical.length] },
      itemStyle: { color: categorical[i % categorical.length] },
      areaStyle: opts.area ? { color: categorical[i % categorical.length], opacity: 0.1 } : undefined,
      connectNulls: false,
    })),
  }
}

/** OHLC candlestick with a volume sub-panel, buy/sell trade markers, and an
 * optional roll-boundary overlay -- a thin wrapper over `superChartOption`
 * for the callers (ProductForm's preview, ProductDrawer.test.tsx) that only
 * ever wanted price+volume with no run overlay. Chinese-market convention:
 * red = up, green = down (theme/palette.ts). */
export function candlestickOption(
  dark: boolean,
  bars: Bar[],
  opts: {
    signals?: { date: string; direction: string }[]
    roll?: RollPoint[]
  } = {},
): EChartsOption {
  return superChartOption({ dark, bars, panes: ['volume'], signals: opts.signals, roll: opts.roll })
}
