import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { productsApi } from '../api/endpoints'
import type { Product } from '../api/types'
import { Table } from '../components/Table'
import type { Column } from '../components/Table'
import { ProductDrawer } from './products/ProductDrawer'

export function ProductsPage() {
  const { t, i18n } = useTranslation()
  const isZh = i18n.resolvedLanguage === 'zh'
  const queryClient = useQueryClient()
  const [drawerCode, setDrawerCode] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const { data: products, isLoading, error } = useQuery({
    queryKey: ['products'],
    queryFn: productsApi.list,
  })

  const closeDrawer = () => {
    setDrawerCode(null)
    setCreating(false)
    void queryClient.invalidateQueries({ queryKey: ['products'] })
  }

  const columns: Column<Product>[] = [
    { key: 'code', header: t('products.code'), render: (p) => <strong>{p.code}</strong> },
    { key: 'name', header: isZh ? t('products.nameZh') : t('products.name'), render: (p) => (isZh ? p.name_zh : p.name) },
    { key: 'exchange', header: t('products.exchange'), render: (p) => p.exchange },
    { key: 'multiplier', header: t('products.multiplier'), render: (p) => p.costs.multiplier },
    { key: 'tick_size', header: t('products.tickSize'), render: (p) => p.tick_size },
    { key: 'margin_rate', header: t('products.marginRate'), render: (p) => `${(p.costs.margin_rate * 100).toFixed(1)}%` },
    {
      key: 'commission',
      header: t('products.commissionMode'),
      render: (p) =>
        p.costs.commission_mode === 'rate'
          ? `${((p.costs.commission_rate ?? 0) * 10000).toFixed(2)}‱`
          : `${(p.costs.commission_per_lot ?? 0).toFixed(2)}${t('products.perLot')}`,
    },
    {
      key: 'main_months',
      header: t('products.mainMonths'),
      render: (p) => p.roll.main_months.map((m) => String(m).padStart(2, '0')).join('/'),
    },
    {
      key: 'coverage',
      header: t('products.coverage'),
      render: (p) =>
        p.coverage.has_data ? (
          <span>
            {p.coverage.n_rows} {t('products.rows')}
            {p.coverage.stale_keys.length > 0 && <span className="badge badge-warning" style={{ marginLeft: 6 }}>{t('data.stale')}</span>}
          </span>
        ) : (
          <span className="badge badge-neutral">{t('products.noData')}</span>
        ),
    },
  ]

  return (
    <div className="page">
      <div className="page-header">
        <div className="toolbar">
          <div>
            <h1>{t('products.title')}</h1>
            <p>{t('products.subtitle')}</p>
          </div>
          <div className="spacer" />
          <button className="btn btn-primary" onClick={() => setCreating(true)}>
            + {t('products.addProduct')}
          </button>
        </div>
      </div>

      {error && <div className="hint-banner warning">{errorMessage(error)}</div>}

      {isLoading ? (
        <div className="empty-state">{t('common.loading')}</div>
      ) : (
        <Table
          columns={columns}
          rows={products ?? []}
          rowKey={(p) => p.code}
          onRowClick={(p) => setDrawerCode(p.code)}
        />
      )}

      {(drawerCode || creating) && (
        <ProductDrawer code={drawerCode} onClose={closeDrawer} />
      )}
    </div>
  )
}
