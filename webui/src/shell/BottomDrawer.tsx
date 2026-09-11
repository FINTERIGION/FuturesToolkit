import { useTranslation } from 'react-i18next'
import type { TabItem } from '../components/Tabs'
import { Tabs } from '../components/Tabs'
import { BacktestPanel } from '../panels/BacktestPanel'
import { HistoryPanel } from '../panels/HistoryPanel'
import { useDrawerResize } from './useDrawerResize'
import type { DrawerTab } from './WorkspaceContext'
import { useWorkspace } from './WorkspaceContext'

/**
 * Not a variant of `components/Drawer` -- that one is a modal, right-side
 * portal with a backdrop, and the bottom drawer must never be modal: the
 * chart above it stays interactive (zoomable, hoverable) while this is
 * open. Backtest and History are tabs here rather than routes, so switching
 * between them no longer costs whatever the previous tab's page held in
 * state -- notably a live job's SSE stream, now held one level up in
 * `JobsProvider` instead of inside whichever panel started it.
 */
export function BottomDrawer() {
  const { t } = useTranslation()
  const { drawerTab, setDrawerTab, drawerCollapsed, setDrawerCollapsed, drawerHeight, setDrawerHeight, backtestPrefill } =
    useWorkspace()
  const { onPointerDown } = useDrawerResize(drawerHeight, setDrawerHeight)

  const tabs: TabItem[] = [
    { key: 'backtest', label: t('workspace.tabBacktest') },
    { key: 'runs', label: t('workspace.tabRuns') },
  ]

  // The toggle button stays in the exact same slot (far right of this row)
  // whether collapsed or not, and just flips via CSS rotation -- so
  // collapsing and expanding read as one reversible action, not two
  // differently-placed controls.
  if (drawerCollapsed) {
    return (
      <div className="bdrawer bdrawer-collapsed">
        <div className="bdrawer-tabbar">
          <span className="bdrawer-collapsed-label">{tabs.find((x) => x.key === drawerTab)?.label}</span>
          <div className="spacer" />
          <button
            className="btn btn-sm btn-ghost bdrawer-toggle is-collapsed"
            onClick={() => setDrawerCollapsed(false)}
            title={t('workspace.expandDrawer')}
          >
            ▾
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="bdrawer" style={{ height: 'var(--drawer-h)' }}>
      <div className="bdrawer-handle" onPointerDown={onPointerDown} title={t('workspace.resizeDrawer')} />
      <div className="bdrawer-tabbar">
        <Tabs items={tabs} active={drawerTab} onChange={(k) => setDrawerTab(k as DrawerTab)} />
        <div className="spacer" />
        <button className="btn btn-sm btn-ghost bdrawer-toggle" onClick={() => setDrawerCollapsed(true)} title={t('workspace.collapseDrawer')}>
          ▾
        </button>
      </div>
      <div className="bdrawer-body">
        {/* Remounted (`key={seq}`) whenever a new prefill arrives -- see
            BacktestPanel's own doc comment for why that has to be a remount
            rather than a prop change an effect reacts to. */}
        {drawerTab === 'backtest' && <BacktestPanel key={backtestPrefill?.seq ?? 0} prefill={backtestPrefill?.value} />}
        {drawerTab === 'runs' && <HistoryPanel />}
      </div>
    </div>
  )
}
