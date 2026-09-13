import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../api/client'
import { jobsApi } from '../api/endpoints'
import type { JobState } from '../api/types'

export interface LogLine {
  level: string
  message: string
}

/** Job statuses that will never change again. Exported because JobProgress
 * needs the same list to decide whether a lost job's last known status was
 * already an answer. */
export const TERMINAL = ['done', 'error', 'cancelled']

/** How often to ask the server directly once the event stream has dropped. */
const POLL_MS = 3000

/** Drives one job's SSE stream (see web/routers/jobs.py) and exposes its live
 * state + accumulated log.
 *
 * The connection itself can drop -- a restarted backend, a sleeping laptop, a
 * proxy timing out. That used to end the job as far as the UI was concerned:
 * `onerror` closed the stream and nothing ever asked again, so a job that had
 * long since finished sat at "running" forever. On error we fall back to
 * polling `jobsApi.get`, which is enough to carry progress and the terminal
 * result even with no stream at all.
 *
 * Polling has to tell two failures apart, though. A network error means the
 * backend may be mid-restart and is worth retrying. A 404 means the job is not
 * coming back: the registry is an in-memory dict (see web/jobs.py), so a
 * restart empties it, and `JobManager._prune` drops finished jobs past
 * `JOB_RETENTION` besides. Retrying that forever polled a job that could never
 * answer, every 3s for the life of the tab, while the UI sat on "running" --
 * and `isActive` stayed true, so the Run button never came back either. A 404
 * stops the poll and raises `lost` instead.
 */
export function useJob() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [state, setState] = useState<JobState | null>(null)
  const [logs, setLogs] = useState<LogLine[]>([])
  const [streaming, setStreaming] = useState(false)
  const [lost, setLost] = useState(false)
  const sourceRef = useRef<EventSource | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const stop = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
    stopPolling()
    setStreaming(false)
  }, [stopPolling])

  /** A stream frame and a plain GET carry the same fields, except that only
   * the GET includes `result`. */
  const applyState = useCallback((payload: Partial<JobState> & { status: string }) => {
    setState(payload as JobState)
  }, [])

  const startPolling = useCallback(
    (id: string) => {
      if (pollRef.current !== null) return
      const tick = () => {
        void jobsApi
          .get(id)
          .then((job) => {
            applyState(job)
            if (TERMINAL.includes(job.status)) stopPolling()
          })
          .catch((err: unknown) => {
            if (err instanceof ApiError && err.status === 404) {
              // The server has no record of this job and never will again.
              // Stop asking and say so; whatever `state` last held is now the
              // most that can be known about it.
              stopPolling()
              setLost(true)
              return
            }
            /* anything else: keep polling, the backend may just be restarting */
          })
      }
      tick()
      pollRef.current = setInterval(tick, POLL_MS)
    },
    [applyState, stopPolling],
  )

  /** Forget the current job outright -- stream closed, state and log dropped.
   * `start` clears the same things before it connects; the workspace's "exit
   * backtest" control calls it on its own, because a *finished* job is what
   * BacktestPanel re-overlays on the chart (see its auto-overlay effect) and
   * leaving one behind made that button undo itself. */
  const reset = useCallback(() => {
    stop()
    setJobId(null)
    setState(null)
    setLogs([])
    setLost(false)
  }, [stop])

  const start = useCallback(
    (id: string) => {
      reset()
      setJobId(id)

      const source = new EventSource(jobsApi.streamUrl(id))
      sourceRef.current = source
      setStreaming(true)

      source.onopen = () => {
        setStreaming(true)
        stopPolling()          // the stream is authoritative again
      }

      source.onmessage = (evt) => {
        try {
          const payload = JSON.parse(evt.data)
          if (payload.type === 'log') {
            setLogs((prev) => [...prev, { level: payload.level, message: payload.message }])
          } else if (payload.type === 'state') {
            applyState(payload)
            if (TERMINAL.includes(payload.status)) {
              source.close()
              sourceRef.current = null
              setStreaming(false)
              // The state frame deliberately excludes `result` (a backtest's
              // can be large). Fetch it once, now that the job is terminal,
              // via the GET that does include it.
              // Failing here costs only the result: the terminal status is
              // already applied, and every panel that renders a run's numbers
              // reads them from the run store rather than from this payload.
              void jobsApi.get(id).then(applyState).catch(() => {})
            }
          }
        } catch {
          // ignore malformed frame
        }
      }

      source.onerror = () => {
        // EventSource retries on its own, but a reconnect replays this job's
        // whole buffered log and would duplicate everything already shown.
        // Close it and switch to polling, which carries status, progress and
        // the final result -- everything except live log lines.
        source.close()
        sourceRef.current = null
        setStreaming(false)
        startPolling(id)
      }
    },
    [applyState, reset, startPolling, stopPolling],
  )

  useEffect(() => stop, [stop])

  const cancel = useCallback(() => {
    if (!jobId) return
    void jobsApi.cancel(jobId).catch(() => {
      // A cancel can legitimately fail: 409 once the job has already reached a
      // terminal state (the click raced the last state frame in), and 404 once
      // the server has forgotten it. Neither is worth an alert -- but leaving
      // the rejection unhandled meant the button did nothing, said nothing,
      // and left the panel showing a job it had just been told is over. Ask
      // for the current state instead, and let that speak.
      void jobsApi
        .get(jobId)
        .then(applyState)
        .catch((err: unknown) => {
          if (err instanceof ApiError && err.status === 404) setLost(true)
        })
    })
  }, [applyState, jobId])

  return {
    jobId,
    state,
    logs,
    start,
    cancel,
    reset,
    /** False once the event stream has dropped and the hook is polling instead
     * -- callers can use it to explain why the log stopped growing. */
    streaming,
    /** The server no longer has this job (it restarted, or the job aged out of
     * `JOB_RETENTION`). Its last known state is all there is; it will not
     * progress and cannot be cancelled. */
    lost,
    // A lost job is not active in any sense the UI cares about -- leaving it
    // "running" is what kept the Run button disabled with nothing to wait for.
    isActive: !lost && state ? ['queued', 'running'].includes(state.status) : false,
  }
}
