import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { dataApi, productsApi } from '../api/endpoints'
import { JobProgress } from '../components/JobProgress'
import { useJobSlot } from './JobsProvider'
import { ProductRow } from './ProductRow'
import { useWorkspace } from './WorkspaceContext'

/**
 * The right-hand product panel: search + list (click to chart, double-click
 * to toggle the backtest universe, coverage badges, per-row download/edit)
 * on top, bulk data download at the bottom. This absorbs what used to be the
 * standalone Products and Data pages -- coverage comes straight off
 * `Product.coverage` (no separate `/data/coverage` fetch), and "Update
 * Selected" acts on the backtest universe.
 */
export function SymbolSidebar() {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const job = useJobSlot('data')
  const [query, setQuery] = useState('')
  const [updateError, setUpdateError] = useState<string | null>(null)
  const { chartSymbol, setChartSymbol, universe, toggleUniverse, setEditingProduct, setSidebarOpen } = useWorkspace()

  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })

  // Seed the chart with a product that actually has data downloaded, once,
  // the first time the catalog loads with nothing picked yet -- the same
  // rule the old BacktestPage used to seed its universe with. Exactly once,
  // so it doesn't fight a symbol the user has already chosen.
  useEffect(() => {
    if (chartSymbol || !products || products.length === 0) return
    const withData = products.filter((p) => p.coverage.has_data)
    setChartSymbol((withData[0] ?? products[0]).code)
  }, [chartSymbol, products, setChartSymbol])

  useEffect(() => {
    if (job.state?.status === 'done') {
      void queryClient.invalidateQueries({ queryKey: ['products'] })
      void queryClient.invalidateQueries({ queryKey: ['product-bars'] })
      void queryClient.invalidateQueries({ queryKey: ['product-roll'] })
    }
  }, [job.state?.status, queryClient])

  const runUpdate = async (symbols: string[]) => {
    // This request really can be refused, and used to be refused in
    // silence: the backend claims each symbol in `_inflight` and answers 409
    // while a download of it is already running (web/routers/data.py).
    setUpdateError(null)
    try {
      const { job_id } = await dataApi.update(symbols, false, false)
      job.start(job_id)
    } catch (err) {
      setUpdateError(errorMessage(err))
    }
  }

  const filtered = (products ?? []).filter((p) => {
    if (!query) return true
    const q = query.toLowerCase()
    return p.code.toLowerCase().includes(q) || p.name.toLowerCase().includes(q) || p.name_zh.includes(query)
  })

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <input
          className="sidebar-search"
          placeholder={t('workspace.searchProducts')}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="btn btn-sm add-product-btn" onClick={() => setEditingProduct('new')} title={t('products.addProduct')}>
          +
        </button>
        <button className="btn btn-sm btn-ghost" onClick={() => setSidebarOpen(false)} title={t('workspace.collapseSidebar')}>
          ☰
        </button>
      </div>
      <div className="sidebar-list" role="listbox">
        {filtered.map((p) => (
          <ProductRow
            key={p.code}
            product={p}
            isCharted={p.code === chartSymbol}
            inUniverse={universe.includes(p.code)}
            onSetChart={() => setChartSymbol(p.code)}
            onToggleUniverse={() => toggleUniverse(p.code)}
            onEdit={() => setEditingProduct(p.code)}
          />
        ))}
        {products && filtered.length === 0 && <div className="empty-state">{t('workspace.noProducts')}</div>}
      </div>
      <div className="sidebar-footer">
        {updateError && (
          <div className="hint-banner warning">
            {t('common.updateFailed')} — {updateError}
          </div>
        )}
        <div className="toolbar">
          <button className="btn btn-sm" onClick={() => void runUpdate(universe)} disabled={universe.length === 0 || job.isActive}>
            {t('data.update')} ({universe.length})
          </button>
          <button
            className="btn btn-sm btn-primary"
            onClick={() => void runUpdate((products ?? []).map((p) => p.code))}
            disabled={job.isActive}
          >
            {t('data.updateAll')}
          </button>
        </div>
        {job.state && (
          <JobProgress state={job.state} logs={job.logs} onCancel={job.cancel} streaming={job.streaming} lost={job.lost} />
        )}
      </div>
    </aside>
  )
}
