import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useLocation, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import { dataApi, optimizeApi, productsApi, strategiesApi } from '../api/endpoints'
import type { OptimizeReport } from '../api/types'
import { foldScoresOption, optimizeProgressOption } from '../charts/builders'
import { EChart } from '../components/EChart'
import { JobProgress } from '../components/JobProgress'
import { SymbolPicker } from '../components/SymbolPicker'
import { useJob } from '../hooks/useJob'
import { useStickyState } from '../hooks/useStickyState'
import { useIsDarkMode } from '../theme/useIsDarkMode'
import { OptimizeReportsList } from './optimize/OptimizeReportsList'
import { sharedFormDefaults } from './sharedFormDefaults'

interface PrefillState {
  strategy?: string
  symbols?: string[]
  start?: string
  end?: string
}

export function OptimizePage() {
  const { t } = useTranslation()
  const dark = useIsDarkMode()
  const job = useJob()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [error, setError] = useState<string | null>(null)

  // "Tune This Strategy" from BacktestPage arrives as router state.
  const location = useLocation()
  const [prefill] = useState<PrefillState | undefined>(() => location.state as PrefillState | undefined)

  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })
  const { data: coverage } = useQuery({ queryKey: ['coverage'], queryFn: dataApi.coverage })

  // Shared with BacktestPage under the same keys -- see `useStickyState`.
  // Both pages take their starting values from `sharedFormDefaults`, so a key
  // they share cannot start from two different dates.
  const [strategyKey, setStrategyKey] = useStickyState('strategy', sharedFormDefaults.strategy)
  const [symbols, setSymbols] = useStickyState<string[]>('symbols', sharedFormDefaults.symbols)
  const [start, setStart] = useStickyState('start', sharedFormDefaults.start)
  const [end, setEnd] = useStickyState('end', sharedFormDefaults.end)
  const [cash, setCash] = useStickyState('cash', sharedFormDefaults.cash)
  const [slippage, setSlippage] = useStickyState('slippage', sharedFormDefaults.slippage)

  // `undefined` while the query is in flight -- only a loaded, all-empty
  // coverage table means there is genuinely nothing to tune against.
  const hasAnyData = coverage === undefined || coverage.some((c) => c.has_data)

  // See BacktestPage: a negative slippage pays the strategy to trade, and a
  // study run on one optimises against that. The API rejects it as well.
  const slippageValid = slippage >= 0

  const [nTrials, setNTrials] = useState(200)
  const [nFolds, setNFolds] = useState(4)
  const [embargo, setEmbargo] = useState(10)
  const [holdoutFrac, setHoldoutFrac] = useState(0.2)
  const [lambdaStd, setLambdaStd] = useState(0.5)
  const [minTradesPerYear, setMinTradesPerYear] = useState(4)
  const [ddCap, setDdCap] = useState(0.35)
  // Mirrors research/objective.py's DEFAULT_SPARSE_PENALTY. Sending it
  // explicitly rather than leaving it null keeps what the form shows and what
  // the study runs the same number; the backend still falls back to its own
  // default for any caller that omits the field.
  const [sparsePenalty, setSparsePenalty] = useState(0.5)
  const [seed, setSeed] = useState(42)
  const [probeSamples, setProbeSamples] = useState(20)
  const [studyName, setStudyName] = useState('')

  // A "Tune This Strategy" prefill beats whatever the sticky fields
  // remembered. Applied once, on mount.
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
    if (strategies && strategies.length > 0 && !strategyKey) setStrategyKey(strategies[0].key)
  }, [strategies, strategyKey, setStrategyKey])

  // Seed the picker with products that actually have data downloaded, exactly
  // once -- see the same effect on BacktestPage.
  const symbolsInitialized = useRef(Boolean(prefill?.symbols?.length || symbols.length))
  useEffect(() => {
    if (symbolsInitialized.current) return
    if (!products || products.length === 0 || coverage === undefined) return
    symbolsInitialized.current = true
    const withData = new Set(coverage.filter((c) => c.has_data).map((c) => c.symbol))
    const preferred = products.filter((p) => withData.has(p.code))
    setSymbols((preferred.length > 0 ? preferred : products).slice(0, 3).map((p) => p.code))
  }, [products, coverage, setSymbols])

  useEffect(() => {
    if (job.state?.status === 'done') {
      void queryClient.invalidateQueries({ queryKey: ['optimize-reports'] })
    }
  }, [job.state?.status, queryClient])

  const runOptimize = async () => {
    setError(null)
    try {
      const { job_id } = await optimizeApi.start({
        strategy: strategyKey,
        symbols,
        start,
        end,
        cash,
        slippage,
        n_trials: nTrials,
        n_folds: nFolds,
        embargo,
        holdout_frac: holdoutFrac,
        lambda_std: lambdaStd,
        min_trades_per_year: minTradesPerYear,
        dd_cap: ddCap,
        sparse_penalty: sparsePenalty,
        param_overrides: {},
        seed,
        probe_samples: probeSamples,
        study_name: studyName || null,
      })
      job.start(job_id)
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  const trialData = job.state?.progress_data ?? []
  const result = job.state?.status === 'done' ? (job.state.result as { report: OptimizeReport } | undefined) : undefined
  const report = result?.report

  return (
    <div className="page">
      <div className="page-header">
        <h1>{t('optimize.title')}</h1>
        <p>{t('optimize.subtitle')}</p>
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

            <div className="form-grid-3">
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
              <div className="field">
                <label>{t('optimize.nTrials')}</label>
                <input type="number" value={nTrials} onChange={(e) => setNTrials(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.nFolds')}</label>
                <input type="number" value={nFolds} onChange={(e) => setNFolds(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.embargo')}</label>
                <input type="number" value={embargo} onChange={(e) => setEmbargo(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.holdoutFrac')}</label>
                <input type="number" step="0.05" value={holdoutFrac} onChange={(e) => setHoldoutFrac(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.lambdaStd')}</label>
                <input type="number" step="0.1" value={lambdaStd} onChange={(e) => setLambdaStd(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.minTradesPerYear')}</label>
                <input type="number" value={minTradesPerYear} onChange={(e) => setMinTradesPerYear(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.ddCap')}</label>
                <input type="number" step="0.05" value={ddCap} onChange={(e) => setDdCap(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.sparsePenalty')}</label>
                <input type="number" step="0.1" min="0" max="1" value={sparsePenalty} onChange={(e) => setSparsePenalty(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.seed')}</label>
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('optimize.probeSamples')}</label>
                <input type="number" value={probeSamples} onChange={(e) => setProbeSamples(Number(e.target.value))} />
              </div>
            </div>
            <div className="field">
              <label>{t('optimize.studyName')}</label>
              <input value={studyName} onChange={(e) => setStudyName(e.target.value)} placeholder="optional" />
            </div>

            {error && <div className="hint-banner warning">{error}</div>}
            {!slippageValid && <div className="hint-banner warning">{t('common.slippageNegative')}</div>}

            <button
              className="btn btn-primary"
              onClick={() => void runOptimize()}
              disabled={job.isActive || !strategyKey || symbols.length === 0 || !slippageValid}
              style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
            >
              {t('optimize.runOptimize')}
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
                  lost={job.lost}
                />
                {trialData.length > 0 && (
                  <EChart
                    option={optimizeProgressOption(dark, trialData.map((d) => d.trial), trialData.map((d) => d.value))}
                    height={220}
                  />
                )}
              </div>
            </div>
          )}

          {report && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-body">
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                  <h3 style={{ margin: 0 }}>{t('optimize.bestParams')}</h3>
                  <div style={{ flex: 1 }} />
                  {/* The job result carries the whole report, so this needs no
                      extra fetch -- unlike the same action in the history list
                      below, which only has a summary row to work from. */}
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={() =>
                      navigate('/backtest', {
                        state: {
                          strategy: report.strategy_key,
                          symbols: report.symbols,
                          start: report.start,
                          end: report.end,
                          cash: report.cash,
                          slippage: report.slippage,
                          params: report.best_params,
                        },
                      })
                    }
                  >
                    {t('backtest.sendToBacktest')}
                  </button>
                </div>
                <pre style={{ fontSize: 12, background: 'var(--surface-2)', padding: 10, borderRadius: 6, overflowX: 'auto' }}>
                  {JSON.stringify(report.best_params, null, 2)}
                </pre>
                <div className="stat-grid" style={{ marginBottom: 16 }}>
                  <div className="stat-tile">
                    <div className="label">{t('optimize.bestValue')}</div>
                    <div className="value">{report.best_value.toFixed(4)}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="label">{t('optimize.pbo')}</div>
                    <div className="value">{report.diagnostics.pbo.pbo?.toFixed(3) ?? t('common.na')}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="label">{t('optimize.dsr')}</div>
                    <div className="value">{report.diagnostics.dsr.dsr?.toFixed(3) ?? t('common.na')}</div>
                  </div>
                  <div className="stat-tile">
                    <div className="label">{t('optimize.isOosDecay')}</div>
                    <div className="value">{report.diagnostics.is_oos_decay.ratio.toFixed(3)}</div>
                  </div>
                </div>
                <h3 style={{ marginBottom: 10 }}>{t('optimize.foldScores')}</h3>
                <EChart option={foldScoresOption(dark, report.fold_train_scores, report.fold_valid_scores)} height={240} />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <h3>{t('optimize.reports')}</h3>
        </div>
        <div className="card-body">
          <OptimizeReportsList />
        </div>
      </div>
    </div>
  )
}
