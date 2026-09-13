import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { backtestApi, runsApi, strategiesApi } from '../api/endpoints'
import type { RunDetail } from '../api/types'
import { JobProgress } from '../components/JobProgress'
import { paramsAreValid } from '../components/ParamEditor'
import { useJobSlot } from '../shell/JobsProvider'
import { useStickyState } from '../hooks/useStickyState'
import { useWorkspace } from '../shell/WorkspaceContext'
import { BacktestForm } from './BacktestForm'
import { ResultTables } from './ResultTables'
import { sharedFormDefaults } from './sharedFormDefaults'

/** What this panel itself reads out of a prefill -- deliberately narrower
 * than `WorkspaceContext.BacktestPrefill`, which also carries `symbols`: the
 * universe is a shell-level concern (`prefillBacktest` sets `chartSymbol` /
 * `extraSymbols` directly), not something this panel seeds. Every field is
 * optional because a prefill applies whatever it names and leaves the rest
 * at their sticky values -- a prefill carrying only a strategy and symbols,
 * for instance, has no cash/slippage/params to name. */
export interface BacktestFieldsPrefill {
  strategy?: string
  start?: string
  end?: string
  cash?: number
  slippage?: number
  params?: Record<string, unknown>
}

/**
 * The drawer's Backtest tab. Ported from the old `BacktestPage` essentially
 * unchanged: the sticky-state fields, the params-owner ref that clears
 * params only on a *real* strategy switch, and the one-shot
 * `useState(() => prefill)` read are all the same load-bearing behavior --
 * only the prefill's source changed, from `useLocation().state` to a prop
 * the shell remounts this component on (`key={seq}` at the call site, see
 * BottomDrawer.tsx). The universe is no longer picked here: it comes from
 * `WorkspaceContext.universe`, the same list the sidebar's checkboxes edit.
 */
export function BacktestPanel({ prefill }: { prefill?: BacktestFieldsPrefill }) {
  const { t } = useTranslation()
  const job = useJobSlot('backtest')
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const { universe, runId, setRunId } = useWorkspace()

  // Read once, the way BacktestPage's router-state prefill was: applying it
  // in an effect instead left one render where the form still showed the
  // *stored* value while it was already meant to show the prefilled one, and
  // an effect keyed on the field then fired on a change that was never a
  // user edit. This component is remounted (`key={seq}`) whenever a new
  // prefill arrives, so "once, in the initializer" is still correct.
  const [initialPrefill] = useState<BacktestFieldsPrefill | undefined>(prefill)

  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })

  const [strategyKey, setStrategyKey] = useStickyState('strategy', sharedFormDefaults.strategy, initialPrefill?.strategy)
  const [start, setStart] = useStickyState('start', sharedFormDefaults.start, initialPrefill?.start)
  const [end, setEnd] = useStickyState('end', sharedFormDefaults.end, initialPrefill?.end)
  const [cash, setCash] = useStickyState('cash', sharedFormDefaults.cash, initialPrefill?.cash)
  const [slippage, setSlippage] = useStickyState('slippage', sharedFormDefaults.slippage, initialPrefill?.slippage)
  const [params, setParams] = useState<Record<string, unknown>>(initialPrefill?.params ?? {})

  const strategy = useMemo(() => strategies?.find((s) => s.key === strategyKey), [strategies, strategyKey])
  const paramsValid = useMemo(
    () => !strategy || paramsAreValid(strategy.params, strategy.space, params),
    [strategy, params],
  )
  // Slippage is a cost and cannot be negative -- a negative one fills every
  // trade better than the market and inflates the whole run. The API rejects
  // it too (web/schemas.py); blocking it here is so the user is told before
  // they wait for a job, and matches how out-of-range params already read.
  const slippageValid = slippage >= 0

  useEffect(() => {
    if (strategies && strategies.length > 0 && !strategyKey) {
      setStrategyKey(strategies[0].key)
    }
  }, [strategies, strategyKey, setStrategyKey])

  // Params belong to one strategy, so switching strategy has to clear them --
  // but only a real switch, not this component being remounted with a
  // prefill for the strategy it already had.
  const paramsOwner = useRef(strategyKey)
  useEffect(() => {
    if (paramsOwner.current === strategyKey) return
    paramsOwner.current = strategyKey
    setParams({})
  }, [strategyKey])

  useEffect(() => {
    if (job.state?.status === 'done') {
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
    }
  }, [job.state?.status, queryClient])

  const runBacktest = async () => {
    setError(null)
    try {
      const { job_id, run_id } = await backtestApi.start({
        strategy: strategyKey,
        symbols: universe,
        start,
        end,
        cash,
        slippage,
        params,
      })
      job.start(job_id)
      // A fresh run gets its own overlay even if the server hands back an id
      // this panel has charted before.
      overlaidRunId.current = null
      setLastRunId(run_id)
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  const [lastRunId, setLastRunId] = useState<string | null>(null)
  /** The run this panel has already put on the chart. Each finished run is
   * overlaid once, by id, rather than re-applied on every render the effect
   * below happens to re-run on: `setRunId` is rebuilt on every URL change
   * (react-router rebuilds `setSearchParams` from the current
   * `searchParams`), so depending on it alone re-fires the effect on
   * navigation that has nothing to do with this run -- including the URL
   * change made by the toolbar's own "exit backtest", which is how that
   * button came to undo itself. */
  const overlaidRunId = useRef<string | null>(null)

  // The job result carries this run's metrics, and `blown_up` among them.
  const jobMetrics = (job.state?.result as { metrics?: Record<string, unknown> } | undefined)?.metrics
  const jobBlownUp = Boolean(jobMetrics?.blown_up)

  // A finished run overlays the chart automatically -- see the effect below --
  // so once the job is done, the chart's own `runId` is what this panel's
  // result table follows too, not a separate "last run I started" id. That
  // keeps a run reopened from the History tab and a run just launched here
  // showing through the exact same path.
  useEffect(() => {
    if (job.state?.status !== 'done' || !lastRunId) return
    if (overlaidRunId.current === lastRunId) return
    overlaidRunId.current = lastRunId
    setRunId(lastRunId)
  }, [job.state?.status, lastRunId, setRunId])

  const { data: run } = useQuery<RunDetail>({
    queryKey: ['run', runId],
    queryFn: () => runsApi.get(runId as string),
    enabled: runId !== null,
  })

  return (
    <div className="panel-split">
      <div className="panel-rail">
        <BacktestForm
          strategies={strategies ?? []}
          strategyKey={strategyKey}
          onStrategyChange={setStrategyKey}
          universe={universe}
          start={start}
          onStartChange={setStart}
          end={end}
          onEndChange={setEnd}
          cash={cash}
          onCashChange={setCash}
          slippage={slippage}
          onSlippageChange={setSlippage}
          slippageValid={slippageValid}
          strategy={strategy}
          params={params}
          onParamsChange={(name, value) => setParams((p) => ({ ...p, [name]: value }))}
        />

        {error && <div className="hint-banner warning">{error}</div>}
        {!paramsValid && <div className="hint-banner warning">{t('strategy.paramsOutOfRange')}</div>}
        {!slippageValid && <div className="hint-banner warning">{t('common.slippageNegative')}</div>}

        <button
          className="btn btn-primary"
          onClick={() => void runBacktest()}
          disabled={job.isActive || !strategyKey || universe.length === 0 || !paramsValid || !slippageValid}
          style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
        >
          {t('backtest.runBacktest')}
        </button>

        {job.state && (
          <div style={{ marginTop: 16 }}>
            <JobProgress
              state={job.state}
              logs={job.logs}
              onCancel={job.cancel}
              streaming={job.streaming}
              blownUp={jobBlownUp}
              lost={job.lost}
            />
          </div>
        )}
      </div>

      <div className="panel-main">{run && <ResultTables run={run} />}</div>
    </div>
  )
}
