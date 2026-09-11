import { useQuery } from '@tanstack/react-query'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { productsApi } from '../api/endpoints'
import { SuperChart } from '../chart/SuperChart'
import { ProductDrawer } from '../products/ProductDrawer'
import { BottomDrawer } from './BottomDrawer'
import { SymbolSidebar } from './SymbolSidebar'
import { useWorkspace } from './WorkspaceContext'
import { WorkspaceTopBar } from './WorkspaceTopBar'

function CollapsedSidebarRail({ onExpand }: { onExpand: () => void }) {
  const { t } = useTranslation()
  return (
    <div className="sidebar-rail">
      <button className="btn btn-sm btn-ghost" onClick={onExpand} title={t('workspace.expandSidebar')}>
        ☰
      </button>
    </div>
  )
}

/**
 * The whole screen: a top bar, the super chart filling the remaining width,
 * a product sidebar on the right, and a resizable tabbed drawer along the
 * bottom. Everything else in this app used to be four routed pages; this is
 * the single composition root that replaces them (see BottomDrawer,
 * SymbolSidebar, chart/SuperChart).
 */
export function ChartWorkspace() {
  const { i18n } = useTranslation()
  const { sidebarOpen, setSidebarOpen, drawerCollapsed, drawerHeight, chartSymbol, editingProduct, setEditingProduct } =
    useWorkspace()
  const { data: products } = useQuery({ queryKey: ['products'], queryFn: productsApi.list })
  const isZh = i18n.resolvedLanguage === 'zh'
  const chartedProduct = products?.find((p) => p.code === chartSymbol)
  const productName = chartedProduct ? (isZh ? chartedProduct.name_zh : chartedProduct.name) : undefined

  // The grid row must actually shrink on collapse, not just the drawer's own
  // content -- otherwise the row stays reserved at the last resized height
  // (drawerHeight) and the chart above never grows into the freed space.
  const style = {
    '--sidebar-w': sidebarOpen ? '300px' : '40px',
    '--drawer-h': drawerCollapsed ? '40px' : `${drawerHeight}px`,
  } as CSSProperties

  return (
    <div className="workspace-root">
      <WorkspaceTopBar productName={productName} />
      <div className="workspace" style={style}>
        <SuperChart productName={productName} />
        {sidebarOpen ? <SymbolSidebar /> : <CollapsedSidebarRail onExpand={() => setSidebarOpen(true)} />}
        <BottomDrawer />
      </div>
      {editingProduct !== null && (
        <ProductDrawer code={editingProduct === 'new' ? null : editingProduct} onClose={() => setEditingProduct(null)} />
      )}
    </div>
  )
}
