import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import type { RunDetail, RunSummary, StrategyInfo } from '../api/types'
import { JobsProvider } from '../shell/JobsProvider'
import { WorkspaceProvider } from '../shell/WorkspaceContext'

/** A client that fails fast and keeps nothing between tests -- retries would
 * turn an intentionally-rejected query into a multi-second wait. */
function testQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
  })
}

/** Render a panel or the whole workspace under the providers the real app
 * gives it: react-query, a router (so `WorkspaceContext`'s URL-backed
 * fields work), `WorkspaceProvider`, and `JobsProvider`. `path` seeds the
 * URL -- the workspace lives entirely at `/`, with state in the query
 * string (`?symbol=...&run=...&tab=...`). */
export function renderWorkspace(ui: ReactElement, opts?: { path?: string }) {
  return render(
    <QueryClientProvider client={testQueryClient()}>
      <MemoryRouter initialEntries={[opts?.path ?? '/?symbol=SA']}>
        <WorkspaceProvider>
          <JobsProvider>{ui}</JobsProvider>
        </WorkspaceProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

export const STRATEGIES: StrategyInfo[] = [
  {
    key: 'double_ma',
    class_name: 'DoubleMaStrategy',
    module: 'strategies.double_ma',
    file: '/repo/strategies/double_ma.py',
    docstring: '',
    params: { fast: 10, slow: 30 },
    fixed_params: [],
    space: { fast: { kind: 'int', low: 2, high: 100 }, slow: { kind: 'int', low: 5, high: 200 } },
    space_error: null,
  },
  {
    key: 'chan_theory',
    class_name: 'ChanTheoryStrategy',
    module: 'strategies.chan_theory',
    file: '/repo/strategies/chan_theory.py',
    docstring: '',
    params: { atr_mult: 2.0 },
    fixed_params: [],
    space: { atr_mult: { kind: 'float', low: 0.5, high: 5 } },
    space_error: null,
  },
]

export const COVERAGE = [
  { symbol: 'SA', has_data: true, n_rows: 1600, first_date: '2018-01-02', last_date: '2026-09-01', last_refresh: null, stale_keys: [] },
  { symbol: 'CF', has_data: true, n_rows: 1600, first_date: '2018-01-02', last_date: '2026-09-01', last_refresh: null, stale_keys: [] },
]

export const PRODUCTS = COVERAGE.map((c) => ({
  code: c.symbol,
  exchange: 'CZCE',
  name: c.symbol,
  name_zh: c.symbol,
  start_year: 2018,
  multiplier: 10,
  tick_size: 1,
  costs: { multiplier: 10, margin_rate: 0.11, commission_mode: 'rate' as const, commission_rate: 0.0001, commission_per_lot: null },
  roll: { main_months: [1, 5, 9], lead_months: 1 },
  coverage: c,
})) as never

export function runRow(id: string, symbols: string[]): RunSummary {
  return {
    id,
    kind: 'backtest',
    created_at: 1_700_000_000,
    strategy: 'DoubleMaStrategy',
    symbols,
    start: '2020-01-01',
    end: '2024-01-01',
    cash: 200000,
    slippage: 0,
    status: 'done',
    params: {},
    metrics: { sharpe_ratio: 1, blown_up: false },
    error: null,
  }
}

export function runDetail(id: string, symbols: string[]): RunDetail {
  return {
    ...runRow(id, symbols),
    equity_records: [],
    trade_logs: [],
    deferred: {},
    symbols_with_price: symbols,
  }
}

export const EMPTY_BARS = { symbol: 'SA', bars: [] }
export const EMPTY_ROLL = { symbol: 'SA', roll: [] }
