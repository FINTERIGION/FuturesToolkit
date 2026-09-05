import * as echarts from 'echarts/core'
import type { EChartsOption } from 'echarts'
import { BarChart, CandlestickChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'
import { useIsDarkMode } from '../theme/useIsDarkMode'

echarts.use([
  BarChart,
  CandlestickChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
])

export function EChart({
  option,
  height = 320,
  onEvents,
}: {
  option: EChartsOption
  height?: number | string
  onEvents?: Record<string, (params: unknown) => void>
}) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)
  const dark = useIsDarkMode()

  useEffect(() => {
    if (!ref.current) return
    // No named ECharts theme: every option this app builds
    // (src/charts/*.ts) already embeds the validated light/dark colors from
    // theme/palette.ts directly (background, text, axis, series), keyed by
    // `dark`. Re-created on theme change since chart options aren't rebuilt
    // by this component -- callers recompute `option` off the same signal.
    const chart = echarts.init(ref.current)
    chartRef.current = chart

    const resize = () => chart.resize()
    const observer = new ResizeObserver(resize)
    observer.observe(ref.current)

    return () => {
      observer.disconnect()
      chart.dispose()
      chartRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    chart.setOption(option, { notMerge: true })
  }, [option])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !onEvents) return
    for (const [name, handler] of Object.entries(onEvents)) {
      chart.on(name, handler)
    }
    return () => {
      for (const name of Object.keys(onEvents)) {
        chart.off(name)
      }
    }
  }, [onEvents])

  return <div ref={ref} style={{ width: '100%', height }} />
}
