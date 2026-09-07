import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useLocation, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { backtestApi, dataApi, productsApi, strategiesApi } from '../api/endpoints'
import { JobProgress } from '../components/JobProgress'
import { ParamEditor, paramsAreValid } from '../components/ParamEditor'
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
  params?: Record<string, unknown>
}

export function BacktestPage() {
  const { t } = useTranslation()
  const job = useJob()
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)

  // "Send to Backtest" from a past optimize report (OptimizeReportsList)
  // arrives as router state, applied once on mount.
  const location = useLocation()
  const [prefill] = useState<PrefillState | undefined>(() => location.state as PrefillState | undefined)

  const navigate = useNavigate()
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })
  const { data: coverage } = useQuery({ queryKey: ['coverage'], queryFn: dataApi.coverage })

  // Shared with OptimizePage under the same keys, so a universe/date range
  // picked on either page carries over to the other. Both pages take their
  // starting values from `sharedFormDefaults` -- one key must not have two.
  const [strategyKey, setStrategyKey] = useStickyState('strategy', prefill?.strategy ?? sharedFormDefaults.strategy)
  const [symbols, setSymbols] = useStickyState<string[]>('symbols', prefill?.symbols ?? sharedFormDefaults.symbols)
  const [start, setStart] = useStickyState('start', prefill?.start ?? sharedFormDefaults.start)
  const [end, setEnd] = useStickyState('end', prefill?.end ?? sharedFormDefaults.end)
  const [cash, setCash] = useStickyState('cash', sharedFormDefaults.cash)
  const [slippage, setSlippage] = useStickyState('slippage', sharedFormDefaults.slippage)
  const [params, setParams] = useState<Record<string, unknown>>(prefill?.params ?? {})

  // `undefined` while the query is in flight -- only a loaded, all-empty
  // coverage table means there is genuinely nothing to backtest.
  const hasAnyData = coverage === undefined || coverage.some((c) => c.has_data)

  const strategy = useMemo(() => strategies?.find((s) => s.key === strategyKey), [strategies, strategyKey])
  const paramsValid = useMemo(
    () => !strategy || paramsAreValid(strategy.params, strategy.space, params),
    [strategy, params],
  )

  // A "Send to Backtest" prefill beats whatever the sticky fields remembered:
  // the user just asked for *these* settings. Applied once, on mount, so it
  // does not fight them editing the form afterwards.
  const prefillApplied = useRef(false)
  useEffect(() => {
    if (!prefill || prefillApplied.current) return
    prefillApplied.current = true
    if (prefill.strategy) setStrategyKey(prefill.strategy)
    if (prefill.symbols?.length) setSymbols(prefill.symbols)
    if (prefill.start) setStart(prefill.start)
    if (prefill.end) setEnd(prefill.end)
  }, [prefill, setStrategyKey, setSymbols, setStart, setEnd])

  useEffect(() => {
    if (strategies && strategies.length > 0 && !strategyKey) {
      setStrategyKey(strategies[0].key)
    }
  }, [strategies, strategyKey, setStrategyKey])

  // Seed the picker with products that actually have data downloaded -- an
  // untouched `products.slice(0, 3)` could be three symbols whose CSVs were
  // never fetched, so the first run a new user tries fails on empty bars.
  // Exactly once, so it doesn't fight them clearing the picker afterwards.
  const symbolsInitialized = useRef(Boolean(prefill?.symbols?.length || symbols.length))
  useEffect(() => {
    if (symbolsInitialized.current) return
    if (!products || products.length === 0 || coverage === undefined) return
    symbolsInitialized.current = true
    const withData = new Set(coverage.filter((c) => c.has_data).map((c) => c.symbol))
    const preferred = products.filter((p) => withData.has(p.code))
    setSymbols((preferred.length > 0 ? preferred : products).slice(0, 3).map((p) => p.code))
  }, [products, coverage, setSymbols])

  const skippedInitialReset = useRef(false)
  useEffect(() => {
    // Skip exactly once: the initial strategyKey may equal `prefill.strategy`,
    // and its params (from an optimize report) should survive that first
    // render. Any later strategy change is a real switch and should clear them.
    if (prefill && !skippedInitialReset.current) {
      skippedInitialReset.current = true
      return
    }
    setParams({})
  }, [strategyKey, prefill])

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
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  const [lastRunId, setLastRunId] = useState<string | null>(null)

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
                <input type="number" step="any" value={slippage} onChange={(e) => setSlippage(Number(e.target.value))} />
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

            <button
              className="btn btn-primary"
              onClick={() => void runBacktest()}
              disabled={job.isActive || !strategyKey || symbols.length === 0 || !paramsValid}
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
                <JobProgress state={job.state} logs={job.logs} onCancel={job.cancel} streaming={job.streaming} />
              </div>
            </div>
          )}
          {job.state?.status === 'done' && lastRunId && <BacktestResults runId={lastRunId} />}
        </div>
      </div>
    </div>
  )
}
