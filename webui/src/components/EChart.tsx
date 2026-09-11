import * as echarts from 'echarts/core'
import type { EChartsOption } from 'echarts'
import { BarChart, CandlestickChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'
import { useIsDarkMode } from '../theme/useIsDarkMode'

echarts.use([
  BarChart,
  CandlestickChart,
  LineChart,
  ScatterChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  TooltipComponent,
  CanvasRenderer,
])

/** How many category ticks the option's (first) x-axis carries. Used only to
 * tell "the same series, redrawn" from "a different symbol/run, a different
 * date span" -- see `preserveDataZoom`. */
function categoryLength(option: EChartsOption): number {
  const xAxis = option.xAxis
  const first = Array.isArray(xAxis) ? xAxis[0] : xAxis
  const data = (first as { data?: unknown[] } | undefined)?.data
  return Array.isArray(data) ? data.length : 0
}

export function EChart({
  option,
  height = '100%',
  className,
  preserveDataZoom = false,
  onEvents,
}: {
  option: EChartsOption
  height?: number | string
  className?: string
  /** Keep the viewer's pan/zoom window across an option rebuild, as long as
   * the x-axis has the same number of categories as last time.
   * `setOption(..., { notMerge: true })` below resets the zoom to full view
   * on every rebuild -- without this, toggling a chart pane, ticking a
   * sidebar checkbox, or a live SSE progress frame would snap a zoomed-in
   * super chart back out. A changed category count (a different symbol, a
   * different date range) is treated as a new chart and is *not* restored:
   * reapplying a stale window onto a shorter series would zoom into
   * nothing. The remembered window lives in a ref owned by this component,
   * not the chart instance, so it survives the dispose+re-init that a dark
   * mode flip causes below. */
  preserveDataZoom?: boolean
  onEvents?: Record<string, (params: unknown) => void>
}) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)
  const dark = useIsDarkMode()
  const zoomRef = useRef<{ start: number; end: number; len: number } | null>(null)
  const optionRef = useRef(option)

  useEffect(() => {
    if (!ref.current) return
    // No named ECharts theme: every option this app builds
    // (src/charts/*.ts) already embeds the validated light/dark colors from
    // theme/palette.ts directly (background, text, axis, series), keyed by
    // `dark`. Re-created on theme change since chart options aren't rebuilt
    // by this component -- callers recompute `option` off the same signal.
    const chart = echarts.init(ref.current)
    chartRef.current = chart

    // Coalesced into one `resize()` per animation frame: the bottom
    // drawer's drag-to-resize handle changes this container's height on
    // every pointermove, and a full candlestick re-layout on each of those
    // is both wasted work and the usual source of ResizeObserver's own
    // "loop completed with undelivered notifications" console warning.
    let rafId: number | null = null
    const observer = new ResizeObserver(() => {
      if (rafId !== null) return
      rafId = requestAnimationFrame(() => {
        rafId = null
        chart.resize()
      })
    })
    observer.observe(ref.current)

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId)
      observer.disconnect()
      chart.dispose()
      chartRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    optionRef.current = option
    chart.setOption(option, { notMerge: true })
    if (preserveDataZoom) {
      const len = categoryLength(option)
      const saved = zoomRef.current
      if (saved && saved.len === len) {
        chart.dispatchAction({ type: 'dataZoom', start: saved.start, end: saved.end })
      } else {
        zoomRef.current = { start: 0, end: 100, len }
      }
    }
  }, [option, preserveDataZoom])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !preserveDataZoom) return
    // Reads `optionRef.current` rather than closing over `option` directly,
    // so this listener (attached once per chart instance) always measures
    // the axis that is actually on screen right now, not the one from
    // whichever render first wired it up.
    const onZoom = () => {
      const zoom = (chart.getOption().dataZoom as Array<{ start: number; end: number }> | undefined)?.[0]
      if (zoom) zoomRef.current = { start: zoom.start, end: zoom.end, len: categoryLength(optionRef.current) }
    }
    chart.on('datazoom', onZoom)
    return () => {
      chart.off('datazoom', onZoom)
    }
  }, [preserveDataZoom, dark])

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

  return <div ref={ref} className={className} style={{ width: '100%', height }} />
}
