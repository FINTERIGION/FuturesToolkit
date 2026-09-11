/**
 * Starting values for BacktestPanel's run-form fields, persisted under
 * `useStickyState` keys.
 *
 * The universe itself is not part of this: it lives on `WorkspaceContext` as
 * `chartSymbol` + `extraSymbols`, one shared source the panel reads, rather
 * than a sticky key a form could drift from.
 */
export const sharedFormDefaults = {
  strategy: '',
  start: '2018-01-01',
  end: '2026-12-31',
  cash: 200000,
  slippage: 0,
}
