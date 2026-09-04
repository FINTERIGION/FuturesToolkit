import { useTranslation } from 'react-i18next'
import type { Metrics } from '../api/types'

const HEADLINE_KEYS = [
  'sharpe_ratio',
  'sortino_ratio',
  'calmar_ratio',
  'total_return',
  'annualized_return',
  'max_drawdown',
  'win_rate',
  'profit_factor',
  'n_trades',
  'final_equity',
  'turnover',
  'expectancy',
]

const PERCENT_KEYS = new Set(['total_return', 'annualized_return', 'max_drawdown', 'win_rate', 'annualized_volatility', 'capital_exposure'])
const NEGATIVE_IS_BAD = new Set(['max_drawdown'])

function formatValue(key: string, value: unknown, isInf: boolean, infLabel: string, naLabel: string): string {
  if (isInf) return infLabel
  if (value === null || value === undefined) return naLabel
  if (typeof value !== 'number') return String(value)
  if (PERCENT_KEYS.has(key)) return `${value.toFixed(2)}%`
  if (key === 'final_equity' || key === 'expectancy') {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  }
  return value.toFixed(4).replace(/\.?0+$/, '') || '0'
}

export function MetricStats({ metrics, keys = HEADLINE_KEYS }: { metrics: Metrics; keys?: string[] }) {
  const { t } = useTranslation()
  return (
    <div className="stat-grid">
      {keys.map((key) => {
        const raw = metrics[key]
        const isInf = Boolean(metrics[`${key}_is_inf`])
        const text = formatValue(key, raw, isInf, t('common.infinity'), t('common.na'))
        const numeric = typeof raw === 'number' ? raw : null
        const goodDirection = NEGATIVE_IS_BAD.has(key) ? -1 : 1
        const cls =
          numeric !== null && numeric !== 0
            ? (numeric * goodDirection > 0 ? 'positive' : 'negative')
            : ''
        return (
          <div className="stat-tile" key={key}>
            <div className="label">{t(`backtest.metrics.${key}`, key)}</div>
            <div className={`value ${cls}`}>{text}</div>
          </div>
        )
      })}
    </div>
  )
}
