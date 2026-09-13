import { useTranslation } from 'react-i18next'
import type { PaneKey } from '../charts/superChartOption'
import { useJobSlot } from '../shell/JobsProvider'
import { useWorkspace } from '../shell/WorkspaceContext'

const TOGGLABLE: { key: PaneKey; labelKey: string }[] = [{ key: 'volume', labelKey: 'workspace.paneVolume' }]

export function ChartToolbar({ productName }: { productName?: string }) {
  const { t } = useTranslation()
  const { chartSymbol, panes, setPanes, runId, setRunId } = useWorkspace()
  const job = useJobSlot('backtest')

  const togglePane = (key: PaneKey) => {
    setPanes(panes.includes(key) ? panes.filter((p) => p !== key) : [...panes, key])
  }

  /** Leave the backtest view entirely: the run comes off the chart, and the
   * drawer's result tables and finished-job block go with it. Dropping the
   * job is what makes the button hold. BacktestPanel puts a *done* job's run
   * back on the chart on its own (that is how a run just launched gets
   * overlaid at all), so clearing only the `run` URL param left a finished
   * job behind to re-apply it -- the button looked dead. A job still running
   * is left alone: it has a result on the way and overlays it when it lands.
   */
  const exitBacktest = () => {
    setRunId(null)
    if (!job.isActive) job.reset()
  }

  return (
    <div className="chart-toolbar">
      {chartSymbol && (
        <span className="chart-symbol-title">
          <strong>{chartSymbol}</strong>
          {productName && <span className="chart-symbol-name">{productName}</span>}
        </span>
      )}
      <div className="spacer" />
      {TOGGLABLE.map((p) => (
        <button
          key={p.key}
          className={`btn btn-sm ${panes.includes(p.key) ? 'btn-primary' : ''}`}
          onClick={() => togglePane(p.key)}
        >
          {t(p.labelKey)}
        </button>
      ))}
      {runId && (
        <button className="btn btn-sm" onClick={exitBacktest}>
          {t('workspace.exitBacktest')}
        </button>
      )}
    </div>
  )
}
