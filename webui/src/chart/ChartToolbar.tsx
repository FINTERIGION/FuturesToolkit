import { useTranslation } from 'react-i18next'
import type { PaneKey } from '../charts/superChartOption'
import { useWorkspace } from '../shell/WorkspaceContext'

const TOGGLABLE: { key: PaneKey; labelKey: string }[] = [{ key: 'volume', labelKey: 'workspace.paneVolume' }]

export function ChartToolbar({ productName }: { productName?: string }) {
  const { t } = useTranslation()
  const { chartSymbol, panes, setPanes, runId, setRunId } = useWorkspace()

  const togglePane = (key: PaneKey) => {
    setPanes(panes.includes(key) ? panes.filter((p) => p !== key) : [...panes, key])
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
        <button className="btn btn-sm" onClick={() => setRunId(null)}>
          {t('workspace.clearOverlay')}
        </button>
      )}
    </div>
  )
}
