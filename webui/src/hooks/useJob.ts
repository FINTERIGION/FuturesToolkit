import { useCallback, useEffect, useRef, useState } from 'react'
import { jobsApi } from '../api/endpoints'
import type { JobState } from '../api/types'

export interface LogLine {
  level: string
  message: string
}

type TrialPoint = { trial: number; value: number }

const TERMINAL = ['done', 'error', 'cancelled']

/** How often to ask the server directly once the event stream has dropped. */
const POLL_MS = 3000

/** Drives one job's SSE stream (see web/routers/jobs.py) and exposes its live
 * state + accumulated log.
 *
 * Two things the stream does not hand over whole:
 *
 * - `progress_data` arrives as deltas. A study appends one entry per trial and
 *   never trims, so re-sending the list on every frame was quadratic in trial
 *   count. Accumulated here, so consumers still read a complete
 *   `state.progress_data` and know nothing about the wire format.
 * - The connection itself can drop -- a restarted backend, a sleeping laptop,
 *   a proxy timing out. That used to end the job as far as the UI was
 *   concerned: `onerror` closed the stream and nothing ever asked again, so a
 *   job that had long since finished sat at "running" forever. On error we
 *   fall back to polling `jobsApi.get`, which is enough to carry progress and
 *   the terminal result even with no stream at all.
 */
export function useJob() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [state, setState] = useState<JobState | null>(null)
  const [logs, setLogs] = useState<LogLine[]>([])
  const [streaming, setStreaming] = useState(false)
  const sourceRef = useRef<EventSource | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const trialsRef = useRef<TrialPoint[]>([])

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

  /** Merge a server payload with the trial points accumulated so far, so the
   * shape consumers see is the same whether it came from a stream frame
   * (deltas, no `progress_data`) or from a plain GET (the full list). */
  const applyState = useCallback((payload: Partial<JobState> & { status: string }) => {
    if (Array.isArray(payload.progress_data) && payload.progress_data.length >= trialsRef.current.length) {
      trialsRef.current = payload.progress_data
    }
    setState({ ...(payload as JobState), progress_data: trialsRef.current })
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
          .catch(() => {
            /* keep polling: the backend may just be restarting */
          })
      }
      tick()
      pollRef.current = setInterval(tick, POLL_MS)
    },
    [applyState, stopPolling],
  )

  const start = useCallback(
    (id: string) => {
      stop()
      setJobId(id)
      setState(null)
      setLogs([])
      trialsRef.current = []

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
          } else if (payload.type === 'progress_data') {
            trialsRef.current = [...trialsRef.current, ...(payload.entries as TrialPoint[])]
            setState((prev) => (prev ? { ...prev, progress_data: trialsRef.current } : prev))
          } else if (payload.type === 'state') {
            applyState(payload)
            if (TERMINAL.includes(payload.status)) {
              source.close()
              sourceRef.current = null
              setStreaming(false)
              // The state frame deliberately excludes `result` (a backtest or
              // optimize report can be arbitrarily large). Fetch it once, now
              // that the job is terminal, via the GET that does include it.
              void jobsApi.get(id).then(applyState)
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
    [applyState, startPolling, stop, stopPolling],
  )

  useEffect(() => stop, [stop])

  const cancel = useCallback(() => {
    if (jobId) void jobsApi.cancel(jobId)
  }, [jobId])

  return {
    jobId,
    state,
    logs,
    start,
    cancel,
    /** False once the event stream has dropped and the hook is polling instead
     * -- callers can use it to explain why the log stopped growing. */
    streaming,
    isActive: state ? ['queued', 'running'].includes(state.status) : false,
  }
}
