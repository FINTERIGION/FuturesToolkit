import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useLocation, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { backtestApi, dataApi, productsApi, strategiesApi } from '../api/endpoints'
import { JobProgress } from '../components/JobProgress'
import { ParamEditor, paramsAreValid } from '../components/ParamEditor'
import { RunHistory } from '../components/RunHistory'
import { SymbolPicker } from '../components/SymbolPicker'
import { useJob } from '../hooks/useJob'
import { useStickyState } from '../hooks/useStickyState'
import { BacktestResults } from './backtest/BacktestResults'
import { sharedFormDefaults } from './sharedFormDefaults'

interface PrefillState {
  strategy?: string
  symbols?: string[]
  start?: string
  end?: string
  cash?: number
  slippage?: number
  params?: Record<string, unknown>
}

export function BacktestPage() {
  const { t } = useTranslation()
  const job = useJob()
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)

  // "Send to Backtest" from a past optimize report (OptimizeReportsList)
  // arrives as router state, read once so a later re-render cannot re-apply it.
  const location = useLocation()
  const [prefill] = useState<PrefillState | undefined>(() => location.state as PrefillState | undefined)

  const navigate = useNavigate()
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })
  const { data: coverage } = useQuery({ queryKey: ['coverage'], queryFn: dataApi.coverage })

  // Shared with OptimizePage under the same keys, so a universe/date range
  // picked on either page carries over to the other. Both pages take their
  // starting values from `sharedFormDefaults` -- one key must not have two.
  //
  // A "Send to Backtest" prefill beats whatever the sticky fields remembered:
  // the user just asked for *these* settings. It goes in as `useStickyState`'s
  // override rather than through an effect, so the very first render already
  // shows the strategy the report named -- see the params reset below for what
  // the one-render disagreement used to cost.
  const [strategyKey, setStrategyKey] = useStickyState('strategy', sharedFormDefaults.strategy, prefill?.strategy)
  const [symbols, setSymbols] = useStickyState<string[]>(
    'symbols', sharedFormDefaults.symbols, prefill?.symbols?.length ? prefill.symbols : undefined,
  )
  const [start, setStart] = useStickyState('start', sharedFormDefaults.start, prefill?.start)
  const [end, setEnd] = useStickyState('end', sharedFormDefaults.end, prefill?.end)
  // Cash and slippage come across too. They were the one part of a report the
  // hand-off dropped, and dropping them made it a different run: the panel's
  // sticky defaults (200000 / 0) are not what a report tuned under, so the
  // "same" parameters produced numbers that did not match the report they came
  // from, with nothing on screen to say why.
  const [cash, setCash] = useStickyState('cash', sharedFormDefaults.cash, prefill?.cash)
  const [slippage, setSlippage] = useStickyState('slippage', sharedFormDefaults.slippage, prefill?.slippage)
  const [params, setParams] = useState<Record<string, unknown>>(prefill?.params ?? {})

  // `undefined` while the query is in flight -- only a loaded, all-empty
  // coverage table means there is genuinely nothing to backtest.
  const hasAnyData = coverage === undefined || coverage.some((c) => c.has_data)

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

  // Seed the picker with products that actually have data downloaded -- an
  // untouched `products.slice(0, 3)` could be three symbols whose CSVs were
  // never fetched, so the first run a new user tries fails on empty bars.
  // Exactly once, so it doesn't fight them clearing the picker afterwards.
  // A prefill is already in `symbols` by now, so it counts as initialised.
  const symbolsInitialized = useRef(symbols.length > 0)
  useEffect(() => {
    if (symbolsInitialized.current) return
    if (!products || products.length === 0 || coverage === undefined) return
    symbolsInitialized.current = true
    const withData = new Set(coverage.filter((c) => c.has_data).map((c) => c.symbol))
    const preferred = products.filter((p) => withData.has(p.code))
    setSymbols((preferred.length > 0 ? preferred : products).slice(0, 3).map((p) => p.code))
  }, [products, coverage, setSymbols])

  // Params belong to one strategy, so switching strategy has to clear them --
  // but only a real switch. Tracking which strategy the current params were
  // authored for says that directly; the previous "skip the first run" ref
  // could not, because it had no way to tell the initial render from the
  // re-render that applying a prefill caused. With a prefill whose strategy
  // differed from the remembered one -- the ordinary case, since the sticky
  // value is just whatever was tuned last -- this effect ran a second time
  // and wiped the report's `best_params`, and `ParamEditor` then silently
  // fell back to the strategy's defaults. The form looked filled in; the run
  // was not the tuned one. (Under StrictMode's double-invoke it wiped them
  // every time, prefill or not.) Now that the prefill is applied in the
  // initialiser above, `strategyKey` never changes behind the user's back.
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
        symbols,
        start,
        end,
        cash,
        slippage,
        params,
      })
      job.start(job_id)
      setLastRunId(run_id)
      setViewRunId(null)
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  const [lastRunId, setLastRunId] = useState<string | null>(null)

  // The job result carries this run's metrics, and `blown_up` among them (see
  // web/routers/backtest.py, which carries the engine's flag across the empty
  // metrics dict a stopped-on-bar-0 run produces). Read here rather than
  // inside JobProgress, which is shared with Data and Optimize where a job
  // result has no metrics at all.
  const jobMetrics = (job.state?.result as { metrics?: Record<string, unknown> } | undefined)?.metrics
  const jobBlownUp = Boolean(jobMetrics?.blown_up)

  // A run reopened from the history table below. It takes precedence over the
  // run this page just launched, until the user closes it or starts another
  // backtest -- so the results panel always shows the run they last asked for.
  const [viewRunId, setViewRunId] = useState<string | null>(null)
  const finishedRunId = job.state?.status === 'done' ? lastRunId : null
  const activeRunId = viewRunId ?? finishedRunId
  const resultsRef = useRef<HTMLDivElement>(null)

  const openRun = (runId: string) => {
    setViewRunId(runId)
    // The history table sits well below the results panel; without this a
    // row click would look like it did nothing.
    resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  // A deleted run has no artifact left to fetch, so the panel has to let go of
  // it -- via either route it could still be showing: reopened from history,
  // or just produced by this page's own run.
  const forgetRun = (runId: string) => {
    setViewRunId((current) => (current === runId ? null : current))
    setLastRunId((current) => (current === runId ? null : current))
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>{t('backtest.title')}</h1>
        <p>{t('backtest.subtitle')}</p>
      </div>

      {!hasAnyData && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-body">
            <h3 style={{ margin: '0 0 4px' }}>{t('backtest.noData')}</h3>
            <p style={{ margin: '0 0 12px', color: 'var(--text-muted)' }}>{t('backtest.noDataHint')}</p>
            <button className="btn btn-primary" onClick={() => navigate('/data')}>
              {t('backtest.goToData')}
            </button>
          </div>
        </div>
      )}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="card">
          <div className="card-body">
            <div className="field">
              <label>{t('common.strategy')}</label>
              <select value={strategyKey} onChange={(e) => setStrategyKey(e.target.value)}>
                {(strategies ?? []).map((s) => (
                  <option key={s.key} value={s.key}>
                    {s.key}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>{t('common.symbols')}</label>
              <SymbolPicker products={products ?? []} selected={symbols} onChange={setSymbols} />
            </div>

            <div className="form-grid">
              <div className="field">
                <label>{t('common.start')}</label>
                <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
              </div>
              <div className="field">
                <label>{t('common.end')}</label>
                <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
              </div>
              <div className="field">
                <label>{t('common.cash')}</label>
                <input type="number" value={cash} onChange={(e) => setCash(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('common.slippage')}</label>
                <input
                  type="number"
                  step="any"
                  min="0"
                  className={slippageValid ? undefined : 'invalid'}
                  value={slippage}
                  onChange={(e) => setSlippage(Number(e.target.value))}
                />
              </div>
            </div>

            {strategy && (
              <>
                <h3 style={{ margin: '16px 0 8px', fontSize: 13, color: 'var(--text-muted)' }}>{t('strategy.params')}</h3>
                <ParamEditor
                  defaults={strategy.params}
                  space={strategy.space}
                  fixedParams={strategy.fixed_params}
                  values={params}
                  onChange={(name, value) => setParams((p) => ({ ...p, [name]: value }))}
                />
              </>
            )}

            {error && <div className="hint-banner warning">{error}</div>}
            {!paramsValid && <div className="hint-banner warning">{t('strategy.paramsOutOfRange')}</div>}
            {!slippageValid && <div className="hint-banner warning">{t('common.slippageNegative')}</div>}

            <button
              className="btn btn-primary"
              onClick={() => void runBacktest()}
              disabled={job.isActive || !strategyKey || symbols.length === 0 || !paramsValid || !slippageValid}
              style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
            >
              {t('backtest.runBacktest')}
            </button>
            <button
              className="btn"
              onClick={() => navigate('/optimize', { state: { strategy: strategyKey, symbols, start, end } })}
              disabled={!strategyKey || symbols.length === 0}
              style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
            >
              {t('backtest.tuneThis')}
            </button>
          </div>
        </div>

        <div>
          {job.state && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-body">
                <JobProgress
                  state={job.state}
                  logs={job.logs}
                  onCancel={job.cancel}
                  streaming={job.streaming}
                  blownUp={jobBlownUp}
                  lost={job.lost}
                />
              </div>
            </div>
          )}
          <div ref={resultsRef}>
            {viewRunId && (
              <div className="toolbar" style={{ marginBottom: 8 }}>
                <span className="badge badge-neutral">{t('runs.viewingPastRun')}</span>
                <div className="spacer" />
                <button className="btn btn-sm" onClick={() => setViewRunId(null)}>
                  {t('common.close')}
                </button>
              </div>
            )}
            {/* Keyed by run: BacktestResults holds the selected tab and the
                per-symbol price/stat selection in its own state, and without a
                key React reuses the instance across runs. Opening a run whose
                symbols differ from the last one then kept pointing at a symbol
                it does not have, and the price query 404'd into a blank panel
                with nothing to say why. */}
            {activeRunId && <BacktestResults key={activeRunId} runId={activeRunId} />}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <h3>{t('runs.title')}</h3>
        </div>
        <div className="card-body">
          <RunHistory onOpen={openRun} onDeleted={forgetRun} openRunId={viewRunId} />
        </div>
      </div>
    </div>
  )
}
