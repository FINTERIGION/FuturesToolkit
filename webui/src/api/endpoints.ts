import { api } from './client'
import type {
  Bar,
  CompareResult,
  Coverage,
  ExchangeMeta,
  JobState,
  OptimizeReport,
  OptimizeReportSummary,
  Product,
  ProductInput,
  RollPoint,
  RunDetail,
  RunPrice,
  RunSummary,
  StrategyInfo,
} from './types'

export const productsApi = {
  list: () => api.get<Product[]>('/products'),
  get: (code: string) => api.get<Product>(`/products/${code}`),
  create: (code: string, body: ProductInput) => api.post<Product>(`/products/${encodeURIComponent(code)}`, body),
  update: (code: string, body: ProductInput) => api.put<Product>(`/products/${code}`, body),
  remove: (code: string, purgeData: boolean) =>
    api.del<{ deleted: string; purged_files: string[] }>(`/products/${code}?purge_data=${purgeData}`),
  bars: (code: string, start?: string, end?: string) => {
    const q = new URLSearchParams()
    if (start) q.set('start', start)
    if (end) q.set('end', end)
    const qs = q.toString()
    return api.get<{ symbol: string; bars: Bar[] }>(`/products/${code}/bars${qs ? `?${qs}` : ''}`)
  },
  roll: (code: string, start?: string, end?: string) => {
    const q = new URLSearchParams()
    if (start) q.set('start', start)
    if (end) q.set('end', end)
    const qs = q.toString()
    return api.get<{ symbol: string; roll: RollPoint[] }>(`/products/${code}/roll${qs ? `?${qs}` : ''}`)
  },
  exchanges: () => api.get<ExchangeMeta>('/meta/exchanges'),
}

export const dataApi = {
  coverage: () => api.get<Coverage[]>('/data/coverage'),
  update: (symbols: string[], force: boolean, rebuildOnly: boolean) =>
    api.post<{ job_id: string }>('/data/update', { symbols, force, rebuild_only: rebuildOnly }),
}

export const strategiesApi = {
  list: () => api.get<StrategyInfo[]>('/strategies'),
  get: (key: string) => api.get<StrategyInfo>(`/strategies/${key}`),
  reload: () => api.post<{ reloaded: boolean; strategies: string[] }>('/strategies/reload'),
}

export interface BacktestParams {
  strategy: string
  symbols: string[]
  start: string
  end: string
  cash: number
  slippage: number
  params: Record<string, unknown>
}

export const backtestApi = {
  start: (body: BacktestParams) => api.post<{ job_id: string; run_id: string }>('/backtest', body),
}

export interface OptimizeParams {
  strategy: string
  symbols: string[]
  start: string
  end: string
  cash: number
  slippage: number
  n_trials: number
  n_folds: number
  embargo: number
  holdout_frac: number
  lambda_std: number
  min_trades_per_year: number
  dd_cap: number
  sparse_penalty?: number | null
  param_overrides: Record<string, string>
  seed: number
  probe_samples: number
  study_name?: string | null
}

export const optimizeApi = {
  start: (body: OptimizeParams) => api.post<{ job_id: string; run_id: string }>('/optimize', body),
  reports: () => api.get<OptimizeReportSummary[]>('/optimize/reports'),
  report: (name: string) => api.get<OptimizeReport>(`/optimize/reports/${name}`),
  holdout: (name: string, force = false) =>
    api.post<OptimizeReport>(`/optimize/reports/${name}/holdout?force=${force}`),
}

export const runsApi = {
  list: (kind?: string) => api.get<RunSummary[]>(`/runs${kind ? `?kind=${kind}` : ''}`),
  get: (id: string) => api.get<RunDetail>(`/runs/${id}`),
  price: (id: string, symbol: string) => api.get<RunPrice>(`/runs/${id}/price/${symbol}`),
  compare: (ids: string[]) => api.get<CompareResult>(`/runs/compare?ids=${ids.join(',')}`),
  remove: (id: string) => api.del<{ deleted: string }>(`/runs/${id}`),
}

export const jobsApi = {
  list: () => api.get<JobState[]>('/jobs'),
  get: (id: string) => api.get<JobState>(`/jobs/${id}`),
  cancel: (id: string) => api.post<{ cancelled: string }>(`/jobs/${id}/cancel`),
  streamUrl: (id: string) => `/api/jobs/${id}/stream`,
}
