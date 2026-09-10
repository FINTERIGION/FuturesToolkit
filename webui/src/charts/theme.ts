import type { EChartsOption } from 'echarts'
import { CATEGORICAL_DARK, CATEGORICAL_LIGHT, CHART_SURFACE, INK } from '../theme/palette'

export function chartTokens(dark: boolean) {
  const ink = dark ? INK.dark : INK.light
  return {
    surface: dark ? CHART_SURFACE.dark : CHART_SURFACE.light,
    categorical: dark ? CATEGORICAL_DARK : CATEGORICAL_LIGHT,
    ink,
  }
}

/** Shared chrome: transparent background (the card already paints the
 * surface), hairline recessive grid/axes, text in the muted/secondary ink
 * tokens -- never the series color (marks-and-anatomy.md). */
export function baseOption(dark: boolean): EChartsOption {
  const { categorical, ink } = chartTokens(dark)
  return {
    backgroundColor: 'transparent',
    color: categorical,
    textStyle: { color: ink.secondary, fontFamily: 'inherit' },
    grid: { left: 56, right: 20, top: 28, bottom: 44, containLabel: true },
    axisPointer: { lineStyle: { color: ink.axis } },
    legend: {
      top: 0,
      textStyle: { color: ink.secondary, fontSize: 12 },
      itemWidth: 14,
      itemHeight: 8,
    },
    tooltip: {
      backgroundColor: dark ? '#242530' : '#ffffff',
      borderColor: ink.grid,
      borderWidth: 1,
      textStyle: { color: ink.primary, fontSize: 12 },
      extraCssText: 'box-shadow: 0 4px 16px rgba(0,0,0,0.15); border-radius: 6px;',
    },
  }
}

// `overrides: any` is deliberate: ECharts' axis option type is a large
// discriminated union keyed on the literal `type` field, and callers pass
// partial overrides (`{ gridIndex: 0 }`, `{ scale: true }`, ...) that don't
// belong to any single arm of it. Fighting that union for an internal
// chart-building helper isn't worth it -- the public builder functions in
// builders.ts are still typed to return `EChartsOption`, which is what
// callers (EChart, pages) actually consume.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function valueAxis(dark: boolean, overrides: any = {}) {
  const { ink } = chartTokens(dark)
  // `axisLabel` is merged rather than replaced: callers override only the
  // formatter and still want the themed color/size.
  const { axisLabel, ...rest } = overrides
  return {
    type: 'value' as const,
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { lineStyle: { color: ink.grid } },
    axisLabel: { color: ink.muted, fontSize: 11, ...axisLabel },
    ...rest,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function categoryAxis(dark: boolean, data: unknown[], overrides: any = {}) {
  const { ink } = chartTokens(dark)
  return {
    type: 'category' as const,
    data,
    boundaryGap: false,
    axisLine: { lineStyle: { color: ink.axis } },
    axisTick: { show: false },
    axisLabel: { color: ink.muted, fontSize: 11 },
    splitLine: { show: false },
    ...overrides,
  }
}
