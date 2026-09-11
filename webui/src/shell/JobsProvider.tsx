import { createContext, useContext, type ReactNode } from 'react'
import { useJob } from '../hooks/useJob'

export type JobSlot = 'backtest' | 'data'
export type Job = ReturnType<typeof useJob>

const JobsContext = createContext<Record<JobSlot, Job> | null>(null)

/**
 * Two named, always-mounted `useJob()` instances, so a running backtest
 * keeps streaming while the drawer tab showing it is switched away from and
 * back.
 *
 * `useJob` takes no arguments and is driven imperatively via `start(id)` --
 * nothing binds a given instance to a page or a job kind. Calling it twice
 * here, once per slot, gives two independent EventSource connections that
 * live for as long as this provider does (mounted once, at the top of the
 * workspace, above the bottom drawer), rather than for as long as whichever
 * panel happens to be showing. The hook itself is untouched --
 * `useJob.test.ts`'s cases exercise it exactly as before.
 */
export function JobsProvider({ children }: { children: ReactNode }) {
  const backtest = useJob()
  const data = useJob()
  return <JobsContext.Provider value={{ backtest, data }}>{children}</JobsContext.Provider>
}

export function useJobs(): Record<JobSlot, Job> {
  const ctx = useContext(JobsContext)
  if (!ctx) throw new Error('useJobs must be used within a JobsProvider')
  return ctx
}

export function useJobSlot(slot: JobSlot): Job {
  return useJobs()[slot]
}
