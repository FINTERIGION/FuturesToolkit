import { useTranslation } from 'react-i18next'
import type { JobState } from '../api/types'
import { TERMINAL } from '../hooks/useJob'
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
  blownUp = false,
  lost = false,
}: {
  state: JobState | null
  logs: LogLine[]
  onCancel?: () => void
  /** False once the event stream has dropped and useJob fell back to polling.
   * Status and progress still update; live log lines do not, so say so rather
   * than leaving a log that has simply stopped growing. */
  streaming?: boolean
  /** The engine stopped the run on an insolvent account. The job itself did
   * finish -- nothing threw -- so its status is `done` and the badge was
   * green, which is the one thing this outcome must not look like. Shown in
   * its place rather than beside it: a run that blew up has no result to
   * report, only a reason it has none. */
  blownUp?: boolean
  /** The server no longer has this job, so its status can no longer be
   * followed. Distinct from `!streaming`, which is a connection this hook
   * expects to recover from -- this one will not. */
  lost?: boolean
}) {
  const { t } = useTranslation()
  if (!state) return null

  // Nothing to cancel on a job the server has forgotten -- the request would
  // 404 the same way the status poll just did.
  const canCancel =
    onCancel && !lost && (state.status === 'running' || state.status === 'queued') && !state.cancel_requested
  const showBlownUp = blownUp && state.status === 'done'
  const unresolved = lost && !TERMINAL.includes(state.status)

  return (
    <div className="job-progress">
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <span className={`badge ${showBlownUp ? 'badge-danger' : STATUS_BADGE[state.status] ?? 'badge-neutral'}`}>
          {showBlownUp ? t('backtest.blownUp') : t(`jobs.${state.status}`)}
        </span>
        <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>{state.message}</span>
        {unresolved && <span className="badge badge-danger">{t('jobs.lost')}</span>}
        {!lost && !streaming && (state.status === 'running' || state.status === 'queued') && (
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
      {unresolved && (
        <div className="hint-banner warning" style={{ marginBottom: 10 }}>
          {t('jobs.lostHint')}
        </div>
      )}
      {showBlownUp && (
        <div className="hint-banner danger" style={{ marginBottom: 10 }}>
          {t('backtest.blownUpHint')}
        </div>
      )}
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
