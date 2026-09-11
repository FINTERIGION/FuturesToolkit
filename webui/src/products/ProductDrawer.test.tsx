import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { productsApi } from '../api/endpoints'
import { JobsProvider } from '../shell/JobsProvider'
import { ProductDrawer } from './ProductDrawer'

vi.mock('../api/endpoints', () => ({
  productsApi: {
    get: vi.fn(), update: vi.fn(), create: vi.fn(), remove: vi.fn(),
    exchanges: vi.fn(), bars: vi.fn(), roll: vi.fn(),
  },
  jobsApi: { get: vi.fn(), cancel: vi.fn(), streamUrl: (id: string) => `/api/jobs/${id}/stream` },
}))

vi.mock('../components/EChart', () => ({ EChart: () => <div data-testid="chart" /> }))

function product(margin: number, commissionRate: number) {
  return {
    code: 'SA', exchange: 'CZCE', name: 'Soda Ash', name_zh: '纯碱',
    start_year: 2018, multiplier: 20, tick_size: 1,
    costs: { multiplier: 20, margin_rate: margin, commission_mode: 'rate', commission_rate: commissionRate, commission_per_lot: null },
    roll: { main_months: [1, 5, 9], lead_months: 1 },
    coverage: { symbol: 'SA', has_data: true, n_rows: 10, first_date: null, last_date: null, last_refresh: null, stale_keys: [] },
  }
}

function fieldInput(label: string) {
  const field = screen.getByText(label).closest('.field')!
  return within(field as HTMLElement).getByRole('spinbutton')
}

function renderDrawer() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <JobsProvider>
        <ProductDrawer code="SA" onClose={() => {}} />
      </JobsProvider>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.mocked(productsApi.exchanges).mockResolvedValue({ exchanges: ['CZCE', 'SHFE'], default_main_months: [1, 5, 9], default_roll_lead_months: 1 })
  vi.mocked(productsApi.bars).mockResolvedValue({ symbol: 'SA', bars: [] })
  vi.mocked(productsApi.roll).mockResolvedValue({ symbol: 'SA', roll: [] })
  vi.mocked(productsApi.update).mockResolvedValue(product(0.125, 0.0001) as never)
})

describe('rate display', () => {
  it('shows a fractional margin rate as itself, not rounded to a whole percent', async () => {
    vi.mocked(productsApi.get).mockResolvedValue(product(0.125, 0.0001) as never)
    renderDrawer()
    await waitFor(() => expect(fieldInput('Margin Rate')).toHaveValue(12.5))
  })

  it('strips binary floating-point noise rather than showing 11.000000000000002', async () => {
    vi.mocked(productsApi.get).mockResolvedValue(product(0.11, 0.00001) as never)
    renderDrawer()
    await waitFor(() => expect(fieldInput('Margin Rate')).toHaveValue(11))
    // 1e-5 * 10000 is 0.09999999999999999 before rounding.
    expect(fieldInput('Commission')).toHaveValue(0.1)
  })

  it('round-trips an untouched product back unchanged when saved', async () => {
    // The bug that mattered: open a product, change nothing, press Save, and
    // the margin came back rounded -- 12.5% stored as 13%. The panel is where
    // these get edited and the margin rate sizes every position.
    const user = userEvent.setup()
    vi.mocked(productsApi.get).mockResolvedValue(product(0.125, 0.000123) as never)
    renderDrawer()

    await waitFor(() => expect(fieldInput('Margin Rate')).toHaveValue(12.5))
    await user.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(productsApi.update).toHaveBeenCalled())
    const [, body] = vi.mocked(productsApi.update).mock.calls[0]
    expect(body.margin_rate).toBeCloseTo(0.125, 10)
    expect(body.commission_rate).toBeCloseTo(0.000123, 10)
  })
})
