import { useTranslation } from 'react-i18next'
import type { StrategyInfo } from '../api/types'
import { ParamEditor } from '../components/ParamEditor'

export function BacktestForm({
  strategies,
  strategyKey,
  onStrategyChange,
  universe,
  start,
  onStartChange,
  end,
  onEndChange,
  cash,
  onCashChange,
  slippage,
  onSlippageChange,
  slippageValid,
  strategy,
  params,
  onParamsChange,
}: {
  strategies: StrategyInfo[]
  strategyKey: string
  onStrategyChange: (key: string) => void
  universe: string[]
  start: string
  onStartChange: (v: string) => void
  end: string
  onEndChange: (v: string) => void
  cash: number
  onCashChange: (v: number) => void
  slippage: number
  onSlippageChange: (v: number) => void
  slippageValid: boolean
  strategy: StrategyInfo | undefined
  params: Record<string, unknown>
  onParamsChange: (name: string, value: unknown) => void
}) {
  const { t } = useTranslation()

  return (
    <>
      <div className="field">
        <label>{t('common.strategy')}</label>
        <select value={strategyKey} onChange={(e) => onStrategyChange(e.target.value)}>
          {strategies.map((s) => (
            <option key={s.key} value={s.key}>
              {s.key}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>{t('common.symbols')}</label>
        <div className="toolbar" style={{ flexWrap: 'wrap' }}>
          {universe.length === 0 ? (
            <span className="badge badge-neutral">{t('workspace.noProducts')}</span>
          ) : (
            universe.map((sym, i) => (
              <span key={sym} className={`badge ${i === 0 ? 'badge-accent' : 'badge-neutral'}`}>
                {sym}
              </span>
            ))
          )}
        </div>
        <p className="field-hint">{t('workspace.universeFromSidebar')}</p>
      </div>

      <div className="form-grid">
        <div className="field">
          <label>{t('common.start')}</label>
          <input type="date" value={start} onChange={(e) => onStartChange(e.target.value)} />
        </div>
        <div className="field">
          <label>{t('common.end')}</label>
          <input type="date" value={end} onChange={(e) => onEndChange(e.target.value)} />
        </div>
        <div className="field">
          <label>{t('common.cash')}</label>
          <input type="number" value={cash} onChange={(e) => onCashChange(Number(e.target.value))} />
        </div>
        <div className="field">
          <label>{t('common.slippage')}</label>
          <input
            type="number"
            step="any"
            min="0"
            className={slippageValid ? undefined : 'invalid'}
            value={slippage}
            onChange={(e) => onSlippageChange(Number(e.target.value))}
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
            onChange={onParamsChange}
          />
        </>
      )}
    </>
  )
}
