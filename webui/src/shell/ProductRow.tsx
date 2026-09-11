import { useTranslation } from 'react-i18next'
import type { Product } from '../api/types'

/**
 * One row in the product sidebar. Two independent selections live on the
 * same row and have to read as unmistakably different: a single click
 * toggles the row into/out of the backtest universe (multi-select, shown by
 * `aria-selected` and a lighter border), while a double click switches which
 * product the main chart displays (single-select, shown by the accent
 * highlight). Single-clicking the charted row is a no-op on the universe --
 * the charted product is structurally always in it, so there is no state
 * where toggling it would do anything.
 */
export function ProductRow({
  product,
  isCharted,
  inUniverse,
  onSetChart,
  onToggleUniverse,
  onEdit,
}: {
  product: Product
  isCharted: boolean
  inUniverse: boolean
  onSetChart: () => void
  onToggleUniverse: () => void
  onEdit: () => void
}) {
  const { t, i18n } = useTranslation()
  const isZh = i18n.resolvedLanguage === 'zh'
  const name = isZh ? product.name_zh : product.name
  const cov = product.coverage

  return (
    <div
      className={`product-row${isCharted ? ' is-charted' : ''}${inUniverse ? ' is-in-universe' : ''}`}
      role="option"
      aria-selected={inUniverse}
      title={isCharted ? t('workspace.chartedAlwaysIncluded') : t('workspace.inUniverse')}
      onClick={onToggleUniverse}
      onDoubleClick={onSetChart}
    >
      <span className="code">{product.code}</span>
      <span className="name">{name}</span>
      {!cov.has_data ? (
        <span className="badge badge-neutral">{t('products.noData')}</span>
      ) : cov.stale_keys.length > 0 ? (
        <span className="badge badge-warning" title={`${cov.n_rows} ${t('products.rows')}`}>
          {t('data.stale')}
        </span>
      ) : (
        <span className="badge badge-success" title={`${cov.n_rows} ${t('products.rows')}`}>
          {t('data.fresh')}
        </span>
      )}
      <div className="row-actions" onDoubleClick={(e) => e.stopPropagation()}>
        <button
          className="btn btn-sm btn-ghost"
          title={t('products.editProduct')}
          onClick={(e) => {
            e.stopPropagation()
            onEdit()
          }}
        >
          ✎
        </button>
      </div>
    </div>
  )
}
