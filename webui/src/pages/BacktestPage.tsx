import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useLocation } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { backtestApi, productsApi, strategiesApi } from '../api/endpoints'
import { JobProgress } from '../components/JobProgress'
import { ParamEditor, paramsAreValid } from '../components/ParamEditor'
import { SymbolPicker } from '../components/SymbolPicker'
import { useJob } from '../hooks/useJob'
import { BacktestResults } from './backtest/BacktestResults'

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

  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })

  const [strategyKey, setStrategyKey] = useState(prefill?.strategy ?? '')
  const [symbols, setSymbols] = useState<string[]>(prefill?.symbols ?? [])
  const [start, setStart] = useState(prefill?.start ?? '2020-01-01')
  const [end, setEnd] = useState(prefill?.end ?? '2026-12-31')
  const [cash, setCash] = useState(200000)
  const [slippage, setSlippage] = useState(0)
  const [params, setParams] = useState<Record<string, unknown>>(prefill?.params ?? {})

  const strategy = useMemo(() => strategies?.find((s) => s.key === strategyKey), [strategies, strategyKey])
  const paramsValid = useMemo(
    () => !strategy || paramsAreValid(strategy.params, strategy.space, params),
    [strategy, params],
  )

  useEffect(() => {
    if (strategies && strategies.length > 0 && !strategyKey) {
      setStrategyKey(strategies[0].key)
    }
  }, [strategies, strategyKey])

  // Apply the "pick a few products to start with" default exactly once, so
  // it doesn't fight the user clearing the picker to none afterwards.
  const symbolsInitialized = useRef(Boolean(prefill?.symbols?.length))
  useEffect(() => {
    if (products && products.length > 0 && !symbolsInitialized.current) {
      symbolsInitialized.current = true
      setSymbols(products.slice(0, 3).map((p) => p.code))
    }
  }, [products])

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
