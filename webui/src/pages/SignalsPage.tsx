import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { productsApi, signalsApi, strategiesApi } from '../api/endpoints'
import type { SignalReport, SignalRow } from '../api/types'
import { ParamEditor } from '../components/ParamEditor'
import { JobProgress } from '../components/JobProgress'
import { SymbolPicker } from '../components/SymbolPicker'
import { Table } from '../components/Table'
import type { Column } from '../components/Table'
import { useJob } from '../hooks/useJob'

// A stop/take-profit is only actionable if it sits on a price the exchange
// will accept -- an integer number of ticks. The engine arms these from raw
// distances/prices, so round for display to the nearest tradable level.
function tickDecimals(tick: number): number {
  const s = tick.toString()
  const i = s.indexOf('.')
  return i === -1 ? 0 : s.length - i - 1
}

function formatAtTick(price: number | null | undefined, tick: number | undefined): string {
  if (price == null) return '—'
  if (!tick) return String(price)
  const rounded = Math.round(price / tick) * tick
  return rounded.toFixed(tickDecimals(tick))
}

export function SignalsPage() {
  const { t } = useTranslation()
  const job = useJob()
  const [error, setError] = useState<string | null>(null)

  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })
  const { data: history } = useQuery({ queryKey: ['signal-history'], queryFn: signalsApi.history })

  const [useModel, setUseModel] = useState(false)
  const [modelPath, setModelPath] = useState('')
  const [strategyKey, setStrategyKey] = useState('')
  const [symbols, setSymbols] = useState<string[]>([])
  const [start, setStart] = useState('2015-01-01')
  const [end, setEnd] = useState('2026-12-31')
  const [cash, setCash] = useState<number | ''>('')
  const [slippage, setSlippage] = useState<number | ''>('')
  const [params, setParams] = useState<Record<string, unknown>>({})
  const [viewFile, setViewFile] = useState<string | null>(null)

  const strategy = useMemo(() => strategies?.find((s) => s.key === strategyKey), [strategies, strategyKey])

  const tickBySymbol = useMemo(
    () => new Map((products ?? []).map((p) => [p.code, p.tick_size])),
    [products],
  )

  useEffect(() => {
    if (strategies && strategies.length > 0 && !strategyKey) setStrategyKey(strategies[0].key)
  }, [strategies, strategyKey])

  const runSignals = async () => {
    setError(null)
    try {
      const { job_id } = await signalsApi.start({
        strategy: useModel ? null : strategyKey,
        model: useModel ? modelPath : null,
        symbols: symbols.length ? symbols : null,
        start,
        end,
        cash: cash === '' ? null : cash,
        slippage: slippage === '' ? null : slippage,
        params,
      })
      job.start(job_id)
      setViewFile(null)
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  const { data: viewedReport } = useQuery({
    queryKey: ['signal-detail', viewFile],
    queryFn: () => signalsApi.detail(viewFile as string),
    enabled: !!viewFile,
  })

  const liveReport = !viewFile && job.state?.status === 'done' ? (job.state.result as SignalReport | undefined) : undefined
  const report = viewedReport ?? liveReport

  const columns: Column<SignalRow>[] = [
    { key: 'symbol', header: t('common.symbols'), render: (r) => <strong>{r.symbol}</strong> },
    { key: 'contract', header: t('signals.contract'), render: (r) => r.contract },
    { key: 'current_simulated', header: t('signals.currentSimulated'), render: (r) => r.current_simulated },
    { key: 'target', header: t('signals.target'), render: (r) => r.target },
    { key: 'action', header: t('signals.action'), render: (r) => r.action },
    { key: 'tradable', header: t('signals.tradable'), render: (r) => (r.tradable ? '✓' : '✗') },
    { key: 'stop', header: t('signals.stop'), render: (r) => formatAtTick(r.stop, tickBySymbol.get(r.symbol)) },
    {
      key: 'take_profit',
      header: t('signals.takeProfit'),
      render: (r) => formatAtTick(r.take_profit, tickBySymbol.get(r.symbol)),
    },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <h1>{t('signals.title')}</h1>
        <p>{t('signals.subtitle')}</p>
      </div>

      <div className="hint-banner">{t('signals.caveat1')}</div>
      <div className="hint-banner">{t('signals.caveat2')}</div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="card">
          <div className="card-body">
            <label className="checkbox-row">
              <input type="checkbox" checked={useModel} onChange={(e) => setUseModel(e.target.checked)} />
              {t('signals.useModel')}
            </label>

            {useModel ? (
              <div className="field">
                <label>{t('signals.modelPath')}</label>
                <input value={modelPath} onChange={(e) => setModelPath(e.target.value)} placeholder="models/fintermom.joblib" />
              </div>
            ) : (
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
            )}

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
                <input type="number" value={cash} onChange={(e) => setCash(e.target.value === '' ? '' : Number(e.target.value))} />
              </div>
              <div className="field">
                <label>{t('common.slippage')}</label>
                <input type="number" step="any" value={slippage} onChange={(e) => setSlippage(e.target.value === '' ? '' : Number(e.target.value))} />
              </div>
            </div>

            {!useModel && strategy && (
              <ParamEditor
                defaults={strategy.params}
                space={strategy.space}
                fixedParams={strategy.fixed_params}
                values={params}
                onChange={(name, value) => setParams((p) => ({ ...p, [name]: value }))}
              />
            )}

            {error && <div className="hint-banner warning">{error}</div>}

            <button
              className="btn btn-primary"
              onClick={() => void runSignals()}
              disabled={job.isActive || (useModel ? !modelPath : !strategyKey)}
              style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
            >
              {t('signals.getSignals')}
            </button>
          </div>
        </div>

        <div>
          {job.state && !viewFile && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-body">
                <JobProgress state={job.state} logs={job.logs} onCancel={job.cancel} streaming={job.streaming} />
              </div>
            </div>
          )}

          {report && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-body">
                <div className="toolbar">
                  <span className="badge badge-neutral">{t('signals.asOf')}: {report.as_of}</span>
                  <span className="badge badge-neutral">{t('signals.simulatedEquity')}: {report.simulated_equity.toLocaleString()}</span>
                </div>
                <Table columns={columns} rows={report.signals} rowKey={(r) => r.symbol} />
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <h3>{t('signals.history')}</h3>
            </div>
            <div className="card-body">
              <Table
                columns={[
                  { key: 'as_of', header: t('signals.asOf'), render: (h) => h.as_of },
                  { key: 'strategy', header: t('common.strategy'), render: (h) => h.strategy },
                  { key: 'symbols', header: t('common.symbols'), render: (h) => h.symbols.join(', ') },
                ]}
                rows={history ?? []}
                rowKey={(h) => h.file}
                onRowClick={(h) => setViewFile(h.file)}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
