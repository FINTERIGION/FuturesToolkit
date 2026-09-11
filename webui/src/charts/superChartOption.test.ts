import { describe, expect, it } from 'vitest'
import type { Bar, RollPoint } from '../api/types'
import { superChartOption } from './superChartOption'

function bar(date: string, o: number, h: number, l: number, c: number, v = 100): Bar {
  return { date, open: o, high: h, low: l, close: c, settle: c, oi: 1000, volume: v }
}

const BARS: Bar[] = [
  bar('2024-01-02', 10, 11, 9, 10.5),
  bar('2024-01-03', 10.5, 12, 10, 11.5),
  bar('2024-01-04', 11.5, 12.5, 11, 12),
  bar('2024-01-05', 12, 13, 11.5, 12.8),
]
const DATES = BARS.map((b) => b.date)

function grids(option: ReturnType<typeof superChartOption>) {
  return option.grid as Array<{ top: string; height: string }>
}

describe('superChartOption grid layout', () => {
  it('has one grid per pane, none overlapping, all within the usable height', () => {
    for (const panes of [[], ['volume']] as const) {
      const option = superChartOption({ dark: false, bars: BARS, panes: [...panes] })
      const g = grids(option)
      expect(g).toHaveLength(panes.length + 1) // + the always-present price pane
      let prevBottom = 0
      for (const rect of g) {
        const top = parseFloat(rect.top)
        const height = parseFloat(rect.height)
        expect(top).toBeGreaterThanOrEqual(prevBottom)
        expect(top + height).toBeLessThanOrEqual(100 - 10) // leaves room for the slider + labels
        prevBottom = top + height
      }
    }
  })

  it('drops a pane key it does not recognize, instead of corrupting the layout math', () => {
    // Guards a returning user whose browser still has an older pane
    // (equity/position/drawdown) stored in `ft.panes` from before it was
    // retired -- an unrecognized key must not reach `PANE_WEIGHT` at all.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const option = superChartOption({ dark: false, bars: BARS, panes: ['volume', 'equity'] as any })
    expect(grids(option)).toHaveLength(2) // price + volume only
  })

  it('lists every axis index in both dataZoom entries, for any pane count', () => {
    const option = superChartOption({ dark: false, bars: BARS, panes: ['volume'] })
    const dz = option.dataZoom as Array<{ xAxisIndex: number[] }>
    expect(dz).toHaveLength(2)
    for (const entry of dz) expect(entry.xAxisIndex).toEqual([0, 1])
  })

  it('gives every pane the same category axis, so the linked axisPointer and dataZoom stay in sync', () => {
    const option = superChartOption({ dark: false, bars: BARS, panes: ['volume'] })
    const xAxis = option.xAxis as Array<{ data: string[] }>
    for (const axis of xAxis) expect(axis.data).toEqual(DATES)
  })

  it('only shows date labels on the bottom-most pane', () => {
    const option = superChartOption({ dark: false, bars: BARS, panes: ['volume'] })
    const xAxis = option.xAxis as Array<{ axisLabel: { show: boolean } }>
    expect(xAxis.map((a) => a.axisLabel.show)).toEqual([false, true])
  })
})

describe('superChartOption overlay data', () => {
  it('draws one markLine per contract change, not one per row', () => {
    const roll: RollPoint[] = [
      { date: '2024-01-02', contract: 'SA401', close: 10 },
      { date: '2024-01-03', contract: 'SA401', close: 11 },
      { date: '2024-01-04', contract: 'SA405', close: 12 },
      { date: '2024-01-05', contract: 'SA405', close: 12.5 },
    ]
    const option = superChartOption({ dark: false, bars: BARS, roll })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const price = (option.series as any[])[0]
    expect(price.markLine.data).toEqual([{ xAxis: '2024-01-04' }])
  })

  it('drops a signal whose date has no bar, instead of pinning it to the first bar', () => {
    const option = superChartOption({
      dark: false,
      bars: BARS,
      signals: [
        { date: '2024-01-03', direction: 'buy' },
        { date: '2023-06-01', direction: 'sell' }, // not in BARS
      ],
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const price = (option.series as any[])[0]
    const coords = price.markPoint.data.map((d: { coord: [string, number] }) => d.coord[0])
    expect(coords).toEqual(['2024-01-03'])
  })

  it('clamps a window that starts before the first bar to the first category', () => {
    const option = superChartOption({
      dark: false,
      bars: BARS,
      window: { start: '2020-01-01', end: '2024-01-03' },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const price = (option.series as any[])[0]
    expect(price.markArea.data[0][0].xAxis).toBe('2024-01-02')
    expect(price.markArea.data[0][1].xAxis).toBe('2024-01-03')
  })

  it('renders no markArea when the window falls entirely outside the bars', () => {
    const option = superChartOption({
      dark: false,
      bars: BARS,
      window: { start: '2030-01-01', end: '2030-02-01' },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const price = (option.series as any[])[0]
    expect(price.markArea).toBeUndefined()
  })
})
