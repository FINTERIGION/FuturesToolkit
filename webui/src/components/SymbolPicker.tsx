import { useTranslation } from 'react-i18next'
import type { Product } from '../api/types'

export function SymbolPicker({
  products,
  selected,
  onChange,
}: {
  products: Product[]
  selected: string[]
  onChange: (symbols: string[]) => void
}) {
  const { i18n, t } = useTranslation()
  const isZh = i18n.resolvedLanguage === 'zh'

  const toggle = (code: string) => {
    onChange(selected.includes(code) ? selected.filter((s) => s !== code) : [...selected, code])
  }

  const allSelected = products.length > 0 && products.every((p) => selected.includes(p.code))
  const toggleAll = () => onChange(allSelected ? [] : products.map((p) => p.code))

  return (
    <div className="symbol-picker">
      <div className="symbol-picker-header">
        <button className="btn btn-sm btn-ghost" onClick={toggleAll} disabled={products.length === 0}>
          {allSelected ? t('common.deselectAll') : t('common.selectAll')}
        </button>
      </div>
      <div className="symbol-grid">
        {products.map((p) => (
          <label key={p.code} className={`symbol-chip ${selected.includes(p.code) ? 'selected' : ''}`}>
            <input type="checkbox" checked={selected.includes(p.code)} onChange={() => toggle(p.code)} />
            <span className="symbol-code">{p.code}</span>
            <span className="symbol-name">{isZh ? p.name_zh : p.name}</span>
          </label>
        ))}
      </div>
    </div>
  )
}
