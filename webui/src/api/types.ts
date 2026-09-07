// Shapes mirror what the FastAPI backend actually returns (see web/routers/*.py
// and web/serialize.py on the Python side). Kept loose (optional fields,
// index signatures for metrics) rather than a strict 1:1 mirror, because the
// engine's own metrics dict (core/metrics.py) is the source of truth and this
// layer should not have to change every time a field is added there.

export interface ProductCosts {
  multiplier: number
  margin_rate: number
  commission_mode: 'rate' | 'per_lot'
  commission_rate: number | null
  commission_per_lot: number | null
}

export interface RollRule {
  main_months: number[]
  lead_months: number
}

export interface Coverage {
  symbol: string
  has_data: boolean
  n_rows: number
  first_date: string | null
  last_date: string | null
  last_refresh: string | null
  stale_keys: string[]
}

export interface Product {
  code: string
  exchange: string
  name: string
  name_zh: string
  start_year: number
  multiplier: number
  tick_size: number
  margin_rate?: number
  commission_rate?: number | null
  commission_per_lot?: number | null
  main_months?: number[]
  roll_lead_months?: number
  costs: ProductCosts
  roll: RollRule
  coverage: Coverage
}

export interface ProductInput {
  exchange: string
  name: string
  name_zh: string
  start_year: number
  multiplier: number
  tick_size: number
  margin_rate?: number | null
  commission_rate?: number | null
  commission_per_lot?: number | null
  main_months?: number[] | null
  roll_lead_months?: number | null
}

export interface ExchangeMeta {
  exchanges: string[]
  default_main_months: number[]
  default_roll_lead_months: number
}

export interface Bar {
  date: string
  open: number
  high: number
  low: number
  close: number
  settle: number
  oi: number
  volume: number
}

export interface RollPoint {
  date: string
  contract: string
  close: number
}

// -------------------------------------------------------------------------

export interface SpaceSpec {
  kind: 'int' | 'float' | 'categorical'
  low?: number
  high?: number
  step?: number | null
  log?: boolean
  choices?: unknown[]
}

export interface StrategyInfo {
  key: string
  class_name: string
  module: string
  file: string
  docstring: string
  params: Record<string, unknown>
  fixed_params: string[]
  space: Record<string, SpaceSpec>
  space_error: string | null
}

// -------------------------------------------------------------------------

export type Metrics = Record<string, unknown>

export interface EquityRecord {
  date: string
  equity: number
  daily_return: number
  margin_used: number
  available: number
  position: Record<string, number>
}

export interface TradeLog {
  trade_id: number
  open_date: string
  close_date: string | null
  direction: string
  symbol: string
  contract: string
  contracts: string[]
  n_rolls: number
  open_price: number
  close_price: number | null
  size: number
  gross_pnl: number
  commission: number
  net_pnl: number
  margin_used: number
  open_at_end: boolean
  forced: boolean
  exit_reason: string
  open_bar: number
  close_bar: number | null
}

export interface RunSummary {
  id: string
  kind: string
  created_at: number
  strategy: string | null
  symbols: string[]
  start: string | null
  end: string | null
  cash: number | null
  slippage: number | null
  status: string
  params: Record<string, unknown>
  metrics: Metrics | null
  error: string | null
}

export interface RunDetail extends RunSummary {
  equity_records: EquityRecord[]
  trade_logs: TradeLog[]
  deferred: Record<string, number>
  symbols_with_price: string[]
}

export interface RunPrice {
  symbol: string
  dates: string[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  volume: number[]
  oi: number[]
  signals: Array<{ date: string; price: number; direction: string; size: number; comm: number; symbol: string }>
}

export interface CompareResult {
  runs: RunSummary[]
  equity_curves: Record<string, Array<{ date: string; equity: number }>>
}

// -------------------------------------------------------------------------

export interface JobState {
  id: string
  kind: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  progress: number
  message: string
  error: string | null
  created_at: number
  started_at: number | null
  finished_at: number | null
  cancel_requested: boolean
  /** Present on GET /api/jobs/{id}; the SSE stream sends it as separate
   * delta frames instead, which useJob accumulates back into this field. */
  progress_data: Array<{ trial: number; value: number }>
  result?: unknown
}

export interface OptimizeReportSummary {
  name: string
  strategy: string
  symbols: string[]
  timestamp: string
  best_value: number
  holdout_evaluated: boolean
}

export interface OptimizeReport {
  strategy: string
  strategy_key: string
  symbols: string[]
  start: string
  end: string
  best_params: Record<string, unknown>
  best_value: number
  fold_train_scores: number[]
  fold_valid_scores: number[]
  fold_metrics: Array<{ train: Metrics; valid: Metrics }>
  diagnostics: {
    is_oos_decay: { is_score: number; oos_score: number; ratio: number; warn: boolean }
    pbo: { pbo: number | null; n_combinations: number; n_blocks: number }
    dsr: { dsr: number | null; sr0: number | null; sr_hat: number | null }
    plateau: { base_score: number; dimensions: Record<string, { max_drop_pct: number; flags_spike: boolean }> }
  }
  holdout_evaluated: boolean
  holdout_runs: number
  holdout_metrics?: Metrics
  holdout_score?: number
  [key: string]: unknown
}
