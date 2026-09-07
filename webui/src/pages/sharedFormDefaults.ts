/**
 * Starting values for the run-form fields that BacktestPage and OptimizePage
 * persist under the same `useStickyState` keys.
 *
 * Sharing the keys is deliberate -- see `useStickyState` -- but it only works
 * if both pages agree on where to start. They did not: Backtest declared
 * `start` as 2020-01-01 and Optimize as 2018-01-01, so on a browser with
 * nothing stored yet whichever page was opened first won and wrote its own
 * default for both. Declaring the values once removes that tie-break.
 *
 * 2018 is the shared start date. Optimize splits the range into folds plus a
 * holdout and wants the longer history; a backtest over the wider window is
 * only more history, never a wrong one.
 */
export const sharedFormDefaults = {
  strategy: '',
  symbols: [] as string[],
  start: '2018-01-01',
  end: '2026-12-31',
  cash: 200000,
  slippage: 0,
}
