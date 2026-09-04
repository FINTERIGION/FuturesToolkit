import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { optimizeApi, productsApi, strategiesApi } from '../api/endpoints'
import type { OptimizeReport } from '../api/types'
import { foldScoresOption, optimizeProgressOption } from '../charts/builders'
import { EChart } from '../components/EChart'
import { JobProgress } from '../components/JobProgress'
import { SymbolPicker } from '../components/SymbolPicker'
import { useJob } from '../hooks/useJob'
import { useIsDarkMode } from '../theme/useIsDarkMode'
import { OptimizeReportsList } from './optimize/OptimizeReportsList'

export function OptimizePage() {
  const { t } = useTranslation()
  const dark = useIsDarkMode()
  const job = useJob()
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)

  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })

  const [strategyKey, setStrategyKey] = useState('')
  const [symbols, setSymbols] = useState<string[]>([])
  const [start, setStart] = useState('2018-01-01')
  const [end, setEnd] = useState('2026-12-31')
  const [cash, setCash] = useState(100000)
  const [slippage, setSlippage] = useState(0)
  const [nTrials, setNTrials] = useState(200)
  const [nFolds, setNFolds] = useState(4)
  const [embargo, setEmbargo] = useState(10)
  const [holdoutFrac, setHoldoutFrac] = useState(0.2)
  const [lambdaStd, setLambdaStd] = useState(0.5)
  const [minTradesPerYear, setMinTradesPerYear] = useState(4)
  const [ddCap, setDdCap] = useState(0.35)
  const [seed, setSeed] = useState(42)
  const [probeSamples, setProbeSamples] = useState(20)
  const [studyName, setStudyName] = useState('')

  useEffect(() => {
    if (strategies && strategies.length > 0 && !strategyKey) setStrategyKey(strategies[0].key)
  }, [strategies, strategyKey])

  // Apply the "pick a few products to start with" default exactly once, so
  // it doesn't fight the user clearing the picker to none afterwards.
  const symbolsInitialized = useRef(false)
  useEffect(() => {
    if (products && products.length > 0 && !symbolsInitialized.current) {
      symbolsInitialized.current = true
      setSymbols(products.slice(0, 3).map((p) => p.code))
    }
  }, [products])

  useEffect(() => {
    if (job.state?.status === 'done') {
      void queryClient.invalidateQueries({ queryKey: ['optimize-reports'] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
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
                <input type="number" step="any" value={slippage} onChange={(e) => setSlippage(Number(e.target.value))} />
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

            <button
              className="btn btn-primary"
              onClick={() => void runOptimize()}
              disabled={job.isActive || !strategyKey || symbols.length === 0}
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
                <JobProgress state={job.state} logs={job.logs} onCancel={job.cancel} streaming={job.streaming} />
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
                <h3 style={{ marginBottom: 10 }}>{t('optimize.bestParams')}</h3>
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
