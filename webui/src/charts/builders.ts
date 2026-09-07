import type { EChartsOption } from 'echarts'
import type { Bar, RollPoint } from '../api/types'
import { CANDLE } from '../theme/palette'
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
 * optional roll-boundary overlay (a thin vertical marker at each contract
 * switch). Chinese-market convention: red = up, green = down (theme/palette.ts). */
export function candlestickOption(
  dark: boolean,
  bars: Bar[],
  opts: {
    signals?: { date: string; direction: string }[]
    roll?: RollPoint[]
  } = {},
): EChartsOption {
  const { ink } = chartTokens(dark)
  const dates = bars.map((b) => b.date)
  const ohlc = bars.map((b) => [b.open, b.close, b.low, b.high])
  const volumes = bars.map((b, i) => ({
    value: b.volume,
    itemStyle: { color: b.close >= b.open ? CANDLE.up : CANDLE.down, opacity: 0.5 },
    dataIndex: i,
  }))

  const rollBoundaries: string[] = []
  if (opts.roll && opts.roll.length) {
    let prevContract = opts.roll[0].contract
    for (const point of opts.roll) {
      if (point.contract !== prevContract) {
        rollBoundaries.push(point.date)
        prevContract = point.contract
      }
    }
  }

  const buys = (opts.signals ?? []).filter((s) => s.direction === 'buy').map((s) => s.date)
  const sells = (opts.signals ?? []).filter((s) => s.direction === 'sell').map((s) => s.date)
  const dateIndex = new Map(dates.map((d, i) => [d, i]))
  const priceAt = (d: string) => bars[dateIndex.get(d) ?? 0]?.low ?? 0

  return {
    ...baseOption(dark),
    legend: { show: false },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    tooltip: { ...baseOption(dark).tooltip, trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: [
      { left: 56, right: 20, top: 12, height: '58%', containLabel: true },
      { left: 56, right: 20, top: '72%', height: '16%', containLabel: true },
    ],
    xAxis: [
      categoryAxis(dark, dates, { gridIndex: 0, axisLabel: { show: false } }),
      categoryAxis(dark, dates, { gridIndex: 1 }),
    ],
    yAxis: [
      valueAxis(dark, { gridIndex: 0, scale: true }),
      valueAxis(dark, { gridIndex: 1, splitNumber: 2 }),
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1] },
      { type: 'slider', xAxisIndex: [0, 1], height: 16, bottom: 4, borderColor: 'transparent' },
    ],
    series: [
      {
        type: 'candlestick',
        data: ohlc,
        itemStyle: {
          color: CANDLE.up,
          color0: CANDLE.down,
          borderColor: CANDLE.up,
          borderColor0: CANDLE.down,
        },
        markPoint: {
          symbolSize: 20,
          data: [
            ...buys.map((d) => ({
              name: 'buy',
              coord: [d, priceAt(d)],
              symbol: 'triangle',
              symbolRotate: 0,
              itemStyle: { color: CANDLE.up },
              label: { show: false },
            })),
            ...sells.map((d) => ({
              name: 'sell',
              coord: [d, priceAt(d)],
              symbol: 'triangle',
              symbolRotate: 180,
              itemStyle: { color: CANDLE.down },
              label: { show: false },
            })),
          ],
        },
        markLine: rollBoundaries.length
          ? {
              symbol: 'none',
              silent: true,
              lineStyle: { color: ink.axis, type: 'solid', width: 1, opacity: 0.6 },
              label: { show: false },
              data: rollBoundaries.map((d) => ({ xAxis: d })),
            }
          : undefined,
      },
      {
        type: 'bar',
        data: volumes,
        xAxisIndex: 1,
        yAxisIndex: 1,
        barMaxWidth: 6,
      },
    ],
  }
}

/** Net position over time, per symbol -- a step line since a position is
 * piecewise-constant between fills. */
export function positionOption(dark: boolean, dates: string[], series: { name: string; values: number[] }[]): EChartsOption {
  const { categorical } = chartTokens(dark)
  const showLegend = series.length > 1
  return {
    ...baseOption(dark),
    legend: showLegend ? { ...baseOption(dark).legend, data: series.map((s) => s.name) } : { show: false },
    tooltip: { ...baseOption(dark).tooltip, trigger: 'axis' },
    xAxis: categoryAxis(dark, dates),
    yAxis: valueAxis(dark, { name: '', minInterval: 1 }),
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 16, bottom: 4, borderColor: 'transparent' }],
    series: series.map((s, i) => ({
      name: s.name,
      type: 'line',
      step: 'end',
      data: s.values,
      showSymbol: false,
      lineStyle: { width: 2, color: categorical[i % categorical.length] },
      itemStyle: { color: categorical[i % categorical.length] },
    })),
  }
}

/** Grouped bars: train vs valid score per walk-forward fold. Two series ->
 * always legended (slot 1 = train, slot 2 = valid). */
export function foldScoresOption(dark: boolean, trainScores: number[], validScores: number[]): EChartsOption {
  const { categorical } = chartTokens(dark)
  const folds = trainScores.map((_, i) => `Fold ${i + 1}`)
  return {
    ...baseOption(dark),
    legend: { ...baseOption(dark).legend, data: ['Train', 'Valid'] },
    tooltip: { ...baseOption(dark).tooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: categoryAxis(dark, folds, { boundaryGap: true }),
    yAxis: valueAxis(dark),
    series: [
      { name: 'Train', type: 'bar', data: trainScores, barMaxWidth: 24, itemStyle: { color: categorical[0], borderRadius: [4, 4, 0, 0] } },
      { name: 'Valid', type: 'bar', data: validScores, barMaxWidth: 24, itemStyle: { color: categorical[1], borderRadius: [4, 4, 0, 0] } },
    ],
  }
}

/** Live optimize progress: each trial's objective as a dot, with a running
 * best-so-far line overlaid. Two series (raw trials + running best) always
 * legended. */
export function optimizeProgressOption(dark: boolean, trialNumbers: number[], values: number[]): EChartsOption {
  const { categorical } = chartTokens(dark)
  let best = -Infinity
  const bestSoFar = values.map((v) => {
    best = Math.max(best, v)
    return best
  })
  return {
    ...baseOption(dark),
    legend: { ...baseOption(dark).legend, data: ['Trial', 'Best so far'] },
    tooltip: { ...baseOption(dark).tooltip, trigger: 'axis' },
    xAxis: categoryAxis(dark, trialNumbers.map(String), { boundaryGap: true, name: 'Trial' }),
    yAxis: valueAxis(dark),
    series: [
      {
        name: 'Trial',
        type: 'scatter',
        data: values,
        symbolSize: 8,
        itemStyle: { color: categorical[0] },
      },
      {
        name: 'Best so far',
        type: 'line',
        data: bestSoFar,
        showSymbol: false,
        lineStyle: { width: 2, color: categorical[1] },
        itemStyle: { color: categorical[1] },
      },
    ],
  }
}
