import { useTranslation } from 'react-i18next'
import type { JobState } from '../api/types'
import type { LogLine } from '../hooks/useJob'

const STATUS_BADGE: Record<string, string> = {
  queued: 'badge-neutral',
  running: 'badge-accent',
  done: 'badge-success',
  error: 'badge-danger',
  cancelled: 'badge-warning',
}

export function JobProgress({
  state,
  logs,
  onCancel,
  streaming = true,
}: {
  state: JobState | null
  logs: LogLine[]
  onCancel?: () => void
  /** False once the event stream has dropped and useJob fell back to polling.
   * Status and progress still update; live log lines do not, so say so rather
   * than leaving a log that has simply stopped growing. */
  streaming?: boolean
}) {
  const { t } = useTranslation()
  if (!state) return null

  const canCancel = onCancel && (state.status === 'running' || state.status === 'queued') && !state.cancel_requested

  return (
    <div className="job-progress">
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <span className={`badge ${STATUS_BADGE[state.status] ?? 'badge-neutral'}`}>
          {t(`jobs.${state.status}`)}
        </span>
        <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>{state.message}</span>
        {!streaming && (state.status === 'running' || state.status === 'queued') && (
          <span className="badge badge-warning">{t('jobs.reconnecting')}</span>
        )}
        <div className="spacer" />
        {canCancel && (
          <button className="btn btn-sm btn-danger" onClick={onCancel}>
            {t('jobs.cancel')}
          </button>
        )}
      </div>
      <div className="progress-bar" style={{ marginBottom: 10 }}>
        <div className="fill" style={{ width: `${Math.round(state.progress * 100)}%` }} />
      </div>
      {state.error && (
        <div className="hint-banner warning" style={{ marginBottom: 10 }}>
          {state.error}
        </div>
      )}
      {logs.length > 0 && (
        <div className="job-log">
          {logs.map((line, i) => (
            <div key={i}>{line.message}</div>
          ))}
        </div>
      )}
    </div>
  )
}
