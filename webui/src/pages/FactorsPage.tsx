import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { factorsApi, productsApi } from '../api/endpoints'
import type { FactorCorrJobResult, FactorReportJobResult } from '../api/endpoints'
import { correlationHeatmapOption } from '../charts/builders'
import { EChart } from '../components/EChart'
import { JobProgress } from '../components/JobProgress'
import { ParamEditor } from '../components/ParamEditor'
import { SymbolPicker } from '../components/SymbolPicker'
import { useJob } from '../hooks/useJob'
import { useIsDarkMode } from '../theme/useIsDarkMode'
import { FactorReportView } from './factors/FactorReportView'
import { FactorReportsList } from './factors/FactorReportsList'

export function FactorsPage() {
  const { t } = useTranslation()
  const dark = useIsDarkMode()
  const reportJob = useJob()
  const corrJob = useJob()
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const [corrError, setCorrError] = useState<string | null>(null)

  const { data: factors } = useQuery({ queryKey: ['factors'], queryFn: factorsApi.list })
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })

  const [factorKey, setFactorKey] = useState('')
  const [symbols, setSymbols] = useState<string[]>([])
  const [start, setStart] = useState('2016-01-01')
  const [end, setEnd] = useState('2026-12-31')
  const [horizonsText, setHorizonsText] = useState('1 5 10 20')
  const [returnSource, setReturnSource] = useState('weighted')
  const [nGroups, setNGroups] = useState(3)
  const [paramValues, setParamValues] = useState<Record<string, unknown>>({})
  const [corrHorizon, setCorrHorizon] = useState(5)

  const factor = factors?.find((f) => f.key === factorKey)

  useEffect(() => {
    if (factors && factors.length > 0 && !factorKey) setFactorKey(factors[0].key)
  }, [factors, factorKey])

  // Reset param overrides when the selected factor changes -- carrying
  // over e.g. `lookback` from momentum to carry would silently apply a
  // value the new factor never declared.
  useEffect(() => {
    setParamValues({})
  }, [factorKey])

  const symbolsInitialized = useRef(false)
  useEffect(() => {
    if (products && products.length > 0 && !symbolsInitialized.current) {
      symbolsInitialized.current = true
      setSymbols(products.slice(0, 5).map((p) => p.code))
    }
  }, [products])

  useEffect(() => {
    if (reportJob.state?.status === 'done') {
      void queryClient.invalidateQueries({ queryKey: ['factor-reports'] })
    }
  }, [reportJob.state?.status, queryClient])

  const horizons = horizonsText
    .split(/[\s,]+/)
    .map((s) => parseInt(s, 10))
    .filter((n) => Number.isFinite(n) && n > 0)

  const runReport = async () => {
    setError(null)
    try {
      const { job_id } = await factorsApi.startReport({
        factor: factorKey,
        symbols,
        start,
        end,
        horizons: horizons.length > 0 ? horizons : [1, 5, 10, 20],
        return_source: returnSource,
        n_groups: nGroups,
        params: paramValues,
      })
      reportJob.start(job_id)
    } catch (err) {
      setError(errorMessage(err))
    }
  }

  const runCorr = async () => {
    setCorrError(null)
    try {
      const { job_id } = await factorsApi.startCorr({ symbols, start, end, horizon: corrHorizon })
      corrJob.start(job_id)
    } catch (err) {
      setCorrError(errorMessage(err))
    }
  }

  const reportResult =
    reportJob.state?.status === 'done' ? (reportJob.state.result as FactorReportJobResult | undefined) : undefined
  const corrResult = corrJob.state?.status === 'done' ? (corrJob.state.result as FactorCorrJobResult | undefined) : undefined

  return (
    <div className="page">
      <div className="page-header">
        <h1>{t('factors.title')}</h1>
        <p>{t('factors.subtitle')}</p>
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="card">
          <div className="card-body">
            <div className="field">
              <label>{t('factors.factor')}</label>
              <select value={factorKey} onChange={(e) => setFactorKey(e.target.value)}>
                {(factors ?? []).map((f) => (
                  <option key={f.key} value={f.key}>
                    {f.key} ({f.direction > 0 ? '+' : ''}
                    {f.direction})
                  </option>
                ))}
              </select>
              {factor?.docstring && <span className="field-hint">{factor.docstring.split('\n')[0]}</span>}
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
                <label>{t('factors.horizons')}</label>
                <input value={horizonsText} onChange={(e) => setHorizonsText(e.target.value)} placeholder="1 5 10 20" />
              </div>
              <div className="field">
                <label>{t('factors.returnSource')}</label>
                <select value={returnSource} onChange={(e) => setReturnSource(e.target.value)}>
                  <option value="weighted">{t('factors.sourceWeighted')}</option>
                  <option value="exec">{t('factors.sourceExec')}</option>
                </select>
              </div>
              <div className="field">
                <label>{t('factors.nGroups')}</label>
                <input type="number" min={2} value={nGroups} onChange={(e) => setNGroups(Number(e.target.value))} />
              </div>
            </div>

            {factor && Object.keys(factor.params).length > 0 && (
              <>
                <label style={{ display: 'block', marginTop: 8, marginBottom: 4 }}>{t('factors.factorParams')}</label>
                <ParamEditor
                  defaults={factor.params}
                  space={factor.space}
                  fixedParams={factor.fixed_params}
                  values={paramValues}
                  onChange={(name, value) => setParamValues((prev) => ({ ...prev, [name]: value }))}
                />
              </>
            )}

            {error && <div className="hint-banner warning">{error}</div>}

            <button
              className="btn btn-primary"
              onClick={() => void runReport()}
              disabled={reportJob.isActive || !factorKey || symbols.length === 0}
              style={{ marginTop: 8, width: '100%', justifyContent: 'center' }}
            >
              {t('factors.runReport')}
            </button>
          </div>
        </div>

        <div>
          {reportJob.state && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-body">
                <JobProgress state={reportJob.state} logs={reportJob.logs} onCancel={reportJob.cancel} streaming={reportJob.streaming} />
              </div>
            </div>
          )}
          {reportResult && (
            <div className="card">
              <div className="card-body">
                <FactorReportView report={reportResult.report} />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <h3>{t('factors.correlation')}</h3>
        </div>
        <div className="card-body">
          <p className="field-hint" style={{ marginBottom: 12 }}>
            {t('factors.correlationHint')}
          </p>
          <div className="toolbar" style={{ marginBottom: 12, gap: 12 }}>
            <div className="field" style={{ maxWidth: 160 }}>
              <label>{t('factors.corrHorizon')}</label>
              <input type="number" min={1} value={corrHorizon} onChange={(e) => setCorrHorizon(Number(e.target.value))} />
            </div>
            <button className="btn" onClick={() => void runCorr()} disabled={corrJob.isActive || symbols.length === 0}>
              {t('factors.runCorrelation')}
            </button>
          </div>
          {corrError && <div className="hint-banner warning">{corrError}</div>}
          {corrJob.state && (
            <JobProgress state={corrJob.state} logs={corrJob.logs} onCancel={corrJob.cancel} streaming={corrJob.streaming} />
          )}
          {corrResult && (
            <div className="grid-charts" style={{ marginTop: 16 }}>
              <div>
                <h3 style={{ marginBottom: 10 }}>{t('factors.corrCrossSectional')}</h3>
                <EChart
                  option={correlationHeatmapOption(
                    dark,
                    corrResult.report.correlation_matrix.map((r) => r.factor),
                    corrResult.report.correlation_matrix.map((r) =>
                      corrResult.report.correlation_matrix.map((c) => (r[c.factor] as number | null) ?? null),
                    ),
                  )}
                  height={Math.max(240, corrResult.report.correlation_matrix.length * 40)}
                />
              </div>
              <div>
                <h3 style={{ marginBottom: 10 }}>{t('factors.corrIc')}</h3>
                <EChart
                  option={correlationHeatmapOption(
                    dark,
                    corrResult.report.ic_correlation.map((r) => r.factor),
                    corrResult.report.ic_correlation.map((r) =>
                      corrResult.report.ic_correlation.map((c) => (r[c.factor] as number | null) ?? null),
                    ),
                  )}
                  height={Math.max(240, corrResult.report.ic_correlation.length * 40)}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-header">
          <h3>{t('factors.reports')}</h3>
        </div>
        <div className="card-body">
          <FactorReportsList />
        </div>
      </div>
    </div>
  )
}
