import { useTranslation } from 'react-i18next'
import type { FactorReport } from '../../api/types'
import { annualBarOption, lineSeriesOption } from '../../charts/builders'
import { EChart } from '../../components/EChart'
import type { Column } from '../../components/Table'
import { Table } from '../../components/Table'
import { useIsDarkMode } from '../../theme/useIsDarkMode'

function fmt(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—'
  return n.toFixed(digits)
}

/** Renders one factor report end to end -- shared by the live run on
 * FactorsPage and the "View" drawer on FactorReportsList, so a saved report
 * and a just-finished one render identically. */
export function FactorReportView({ report }: { report: FactorReport }) {
  const { t } = useTranslation()
  const dark = useIsDarkMode()

  const horizonRows = Object.entries(report.ic_decay)
    .map(([h, s]) => ({ horizon: Number(h), ...s }))
    .sort((a, b) => a.horizon - b.horizon)

  const annualRows = Object.entries(report.annual_slices)
    .map(([y, s]) => ({ year: Number(y), ...s }))
    .sort((a, b) => a.year - b.year)

  const groupRows = Object.entries(report.quantiles.groups)
    .map(([g, s]) => ({ group: Number(g), ...s }))
    .sort((a, b) => a.group - b.group)

  const autocorrRows = Object.entries(report.autocorr)
    .map(([lag, v]) => ({ lag: Number(lag), value: v }))
    .sort((a, b) => a.lag - b.lag)

  const icColumns: Column<(typeof horizonRows)[number]>[] = [
    { key: 'horizon', header: t('factors.horizon'), render: (r) => r.horizon },
    { key: 'n_obs', header: t('factors.nObs'), render: (r) => r.n_obs },
    { key: 'mean', header: t('factors.icMean'), render: (r) => fmt(r.mean) },
    { key: 'ir', header: t('factors.icIr'), render: (r) => fmt(r.ir, 3) },
    { key: 't_stat', header: t('factors.tStat'), render: (r) => fmt(r.t_stat, 2) },
    { key: 'p_value', header: t('factors.pValue'), render: (r) => fmt(r.p_value) },
    { key: 'positive_rate', header: t('factors.positiveRate'), render: (r) => `${(r.positive_rate * 100).toFixed(1)}%` },
  ]

  const annualColumns: Column<(typeof annualRows)[number]>[] = [
    { key: 'year', header: t('factors.year'), render: (r) => r.year },
    { key: 'n_bars', header: t('factors.nBars'), render: (r) => r.n_bars },
    { key: 'ic_mean', header: t('factors.icMean'), render: (r) => fmt(r.ic_mean) },
    { key: 'ic_ir', header: t('factors.icIr'), render: (r) => fmt(r.ic_ir, 3) },
    { key: 'long_short_sharpe', header: t('factors.lsSharpe'), render: (r) => fmt(r.long_short_sharpe, 3) },
  ]

  const groupColumns: Column<(typeof groupRows)[number]>[] = [
    { key: 'group', header: t('factors.group'), render: (r) => r.group },
    { key: 'n_obs', header: t('factors.nObs'), render: (r) => r.n_obs },
    { key: 'mean', header: t('factors.mean'), render: (r) => fmt(r.mean, 5) },
    { key: 'sharpe', header: t('factors.sharpe'), render: (r) => fmt(r.sharpe, 3) },
  ]

  const groupSeries = (report.quantile_curve.curves[0] ?? []).map((_, g) => ({
    name: t('factors.groupLabel', { g }),
    values: report.quantile_curve.curves.map((row) => row[g] ?? null),
  }))

  return (
    <div>
      <div className="stat-grid" style={{ marginBottom: 16 }}>
        <div className="stat-tile">
          <div className="label">{t('factors.avgCoverage')}</div>
          <div className="value">
            {report.coverage.mean.toFixed(1)} / {report.coverage.n_symbols}
          </div>
        </div>
        <div className="stat-tile">
          <div className="label">{t('factors.monotonicity')}</div>
          <div className="value">{fmt(report.quantiles.monotonicity, 3)}</div>
        </div>
        <div className="stat-tile">
          <div className="label">{t('factors.turnoverTop')}</div>
          <div className="value">{(report.turnover.top * 100).toFixed(1)}%</div>
        </div>
        <div className="stat-tile">
          <div className="label">{t('factors.turnoverBottom')}</div>
          <div className="value">{(report.turnover.bottom * 100).toFixed(1)}%</div>
        </div>
      </div>

      <h3 style={{ marginBottom: 10 }}>{t('factors.cumulativeIc')}</h3>
      <EChart
        option={lineSeriesOption(dark, report.ic_curve.dates, [
          { name: t('factors.cumulativeIc'), values: report.ic_curve.cumulative_ic },
        ])}
        height={240}
      />

      <h3 style={{ margin: '16px 0 10px' }}>{t('factors.icDecay')}</h3>
      <Table columns={icColumns} rows={horizonRows} rowKey={(r) => String(r.horizon)} />

      <h3 style={{ margin: '16px 0 10px' }}>{t('factors.annualSlices')}</h3>
      <p className="field-hint" style={{ marginBottom: 8 }}>
        {t('factors.annualHint')}
      </p>
      <EChart
        option={annualBarOption(
          dark,
          annualRows.map((r) => String(r.year)),
          annualRows.map((r) => r.ic_mean),
        )}
        height={200}
      />
      <Table columns={annualColumns} rows={annualRows} rowKey={(r) => String(r.year)} />

      <h3 style={{ margin: '16px 0 10px' }}>
        {t('factors.quantileReturns', { n: report.n_groups, h: report.quantiles.horizon })}
      </h3>
      <EChart option={lineSeriesOption(dark, report.quantile_curve.dates, groupSeries)} height={260} />
      <Table columns={groupColumns} rows={groupRows} rowKey={(r) => String(r.group)} />

      <h3 style={{ margin: '16px 0 10px' }}>{t('factors.autocorrelation')}</h3>
      <div className="stat-grid">
        {autocorrRows.map((r) => (
          <div className="stat-tile" key={r.lag}>
            <div className="label">{t('factors.lagLabel', { lag: r.lag })}</div>
            <div className="value">{fmt(r.value, 3)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
