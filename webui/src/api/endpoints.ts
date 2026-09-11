import { api } from './client'
import type {
  Bar,
  Coverage,
  ExchangeMeta,
  JobState,
  Product,
  ProductInput,
  RollPoint,
  RunDetail,
  RunPrice,
  RunSummary,
  StrategyInfo,
} from './types'

/** Every path segment goes through this, not just the ones that happen to
 * carry free-form user text today. Only `create` encoded its code, which left
 * the rule looking like a property of that one call rather than of putting a
 * value in a URL at all -- and `create`'s code is the one the *user types*,
 * so the encoded and unencoded spellings of the same product could disagree.
 * The backend validates codes to letters and digits, so nothing is broken
 * today; this is about the next identifier that is not so tidy. */
const seg = encodeURIComponent

export const productsApi = {
  list: () => api.get<Product[]>('/products'),
  get: (code: string) => api.get<Product>(`/products/${seg(code)}`),
  create: (code: string, body: ProductInput) => api.post<Product>(`/products/${seg(code)}`, body),
  update: (code: string, body: ProductInput) => api.put<Product>(`/products/${seg(code)}`, body),
  remove: (code: string, purgeData: boolean) =>
    api.del<{ deleted: string; purged_files: string[] }>(`/products/${seg(code)}?purge_data=${purgeData}`),
  bars: (code: string, start?: string, end?: string) => {
    const q = new URLSearchParams()
    if (start) q.set('start', start)
    if (end) q.set('end', end)
    const qs = q.toString()
    return api.get<{ symbol: string; bars: Bar[] }>(`/products/${seg(code)}/bars${qs ? `?${qs}` : ''}`)
  },
  roll: (code: string, start?: string, end?: string) => {
    const q = new URLSearchParams()
    if (start) q.set('start', start)
    if (end) q.set('end', end)
    const qs = q.toString()
    return api.get<{ symbol: string; roll: RollPoint[] }>(`/products/${seg(code)}/roll${qs ? `?${qs}` : ''}`)
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
  get: (key: string) => api.get<StrategyInfo>(`/strategies/${seg(key)}`),
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

export const runsApi = {
  list: (kind?: string) => api.get<RunSummary[]>(`/runs${kind ? `?kind=${encodeURIComponent(kind)}` : ''}`),
  get: (id: string) => api.get<RunDetail>(`/runs/${seg(id)}`),
  price: (id: string, symbol: string) => api.get<RunPrice>(`/runs/${seg(id)}/price/${seg(symbol)}`),
  remove: (id: string) => api.del<{ deleted: string }>(`/runs/${seg(id)}`),
}

export const jobsApi = {
  list: () => api.get<JobState[]>('/jobs'),
  get: (id: string) => api.get<JobState>(`/jobs/${seg(id)}`),
  cancel: (id: string) => api.post<{ cancelled: string }>(`/jobs/${seg(id)}/cancel`),
  streamUrl: (id: string) => `/api/jobs/${seg(id)}/stream`,
}
