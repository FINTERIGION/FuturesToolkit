import { useTranslation } from 'react-i18next'
import { useJobs } from './JobsProvider'
import { useWorkspace } from './WorkspaceContext'

const LANGS: { code: string; label: string }[] = [
  { code: 'en', label: 'EN' },
  { code: 'zh', label: '中文' },
]

const PILL_CLASS: Record<string, string> = {
  queued: 'badge-neutral',
  running: 'badge-accent',
}

const SLOT_LABEL_KEY = {
  backtest: 'workspace.tabBacktest',
  data: 'data.title',
} as const

export function WorkspaceTopBar({ productName }: { productName?: string }) {
  const { t, i18n } = useTranslation()
  const { chartSymbol } = useWorkspace()
  const jobs = useJobs()

  const activePills = (['backtest', 'data'] as const)
    .map((slot) => ({ slot, state: jobs[slot].state }))
    .filter((j) => j.state && ['queued', 'running'].includes(j.state.status))

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <span className="brand">FuturesToolkit</span>
        {chartSymbol && (
          <span className="topbar-symbol">
            <strong>{chartSymbol}</strong>
            {productName && <span className="topbar-symbol-name">{productName}</span>}
          </span>
        )}
        <div className="spacer" />
        {activePills.map(({ slot, state }) => (
          <span key={slot} className={`badge ${PILL_CLASS[state!.status] ?? 'badge-neutral'}`} title={state!.message}>
            {t(SLOT_LABEL_KEY[slot])} {Math.round(state!.progress * 100)}%
          </span>
        ))}
        <div className="lang-switch">
          {LANGS.map((l) => (
            <button
              key={l.code}
              className={`lang-btn ${i18n.resolvedLanguage === l.code ? 'active' : ''}`}
              onClick={() => void i18n.changeLanguage(l.code)}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  )
}
