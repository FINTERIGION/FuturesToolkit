const PREFIX = 'ft.'

/**
 * One-shot migration of the old `ft.symbols` sticky key -- the whole
 * universe Backtest and Optimize used to share -- into the workspace's split
 * shape: `ft.chartSymbol` (the main-chart product) plus `ft.extraSymbols`
 * (everything else checked into the backtest universe).
 *
 * Called once, before the app renders (see main.tsx), so `WorkspaceContext`
 * never has to reason about the old key at all. Without this, a returning
 * user's saved universe would silently vanish the moment the workspace
 * shipped -- `ft.chartSymbol`/`ft.extraSymbols` would both read as unset and
 * fall back to their defaults even though `ft.symbols` still held exactly
 * what they had picked.
 *
 * Safe to call on every load: it is a no-op once `ft.chartSymbol` exists,
 * and wrapped like every other `localStorage` access in this app (Safari
 * private mode, a browser blocking site data) so a storage failure never
 * blocks the app from rendering.
 */
export function migrateStickyState(): void {
  try {
    if (localStorage.getItem(PREFIX + 'chartSymbol') !== null) return
    const raw = localStorage.getItem(PREFIX + 'symbols')
    if (raw === null) return
    const symbols = JSON.parse(raw) as unknown
    if (!Array.isArray(symbols) || symbols.length === 0) return
    const [chartSymbol, ...extraSymbols] = symbols as string[]
    localStorage.setItem(PREFIX + 'chartSymbol', JSON.stringify(chartSymbol))
    localStorage.setItem(PREFIX + 'extraSymbols', JSON.stringify(extraSymbols))
    localStorage.removeItem(PREFIX + 'symbols')
  } catch {
    // Storage unavailable or the stored value didn't parse -- the workspace
    // falls back to its own defaults, same as a first-ever visit.
  }
}
