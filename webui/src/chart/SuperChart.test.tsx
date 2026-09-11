import { screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import { productsApi, runsApi, strategiesApi } from '../api/endpoints'
import { PRODUCTS, STRATEGIES, renderWorkspace } from '../test/utils'
import { ChartWorkspace } from '../shell/ChartWorkspace'

vi.mock('../api/endpoints', () => ({
  productsApi: { list: vi.fn(), bars: vi.fn(), roll: vi.fn(), get: vi.fn(), exchanges: vi.fn() },
  dataApi: { update: vi.fn() },
  strategiesApi: { list: vi.fn() },
  backtestApi: { start: vi.fn() },
  runsApi: { list: vi.fn(), get: vi.fn(), price: vi.fn(), remove: vi.fn() },
  jobsApi: { get: vi.fn(), cancel: vi.fn(), streamUrl: (id: string) => `/api/jobs/${id}/stream` },
}))

// Canvas-backed and irrelevant to the states under test.
vi.mock('../components/EChart', () => ({ EChart: () => <div data-testid="chart" /> }))

beforeEach(() => {
  vi.mocked(productsApi.list).mockResolvedValue(PRODUCTS)
  vi.mocked(productsApi.roll).mockRejectedValue(new ApiError(400, 'no roll calendar'))
  vi.mocked(strategiesApi.list).mockResolvedValue(STRATEGIES)
  vi.mocked(runsApi.list).mockResolvedValue([])
})

/** `/products/{code}/bars` 404s for a product whose weighted CSV was never
 * built, which is every product on a fresh install. The chart's loading
 * branch keys off `!barsData`, and a failed query leaves that undefined
 * with `isLoading` already false -- so the state that reads as "still
 * fetching" is the same one a dead query lands in. */
describe('a product with no downloaded data', () => {
  it('says the data is missing instead of loading forever', async () => {
    vi.mocked(productsApi.bars).mockRejectedValue(new ApiError(404, 'Weighted data file not found'))
    renderWorkspace(<ChartWorkspace />, { path: '/?symbol=SA' })

    expect(await screen.findByText('No data downloaded for this product yet')).toBeInTheDocument()
    expect(
      screen.getByText('Download its history from the product sidebar, then come back to see the chart.'),
    ).toBeInTheDocument()
    expect(screen.queryByText('Loading…')).toBeNull()
  })

  it('shows the server’s own message for a failure that is not a missing file', async () => {
    vi.mocked(productsApi.bars).mockRejectedValue(new ApiError(500, 'roll calendar build failed'))
    renderWorkspace(<ChartWorkspace />, { path: '/?symbol=SA' })

    expect(await screen.findByText('roll calendar build failed')).toBeInTheDocument()
    // The download hint would be wrong here: the data is there, the build broke.
    expect(screen.queryByText('No data downloaded for this product yet')).toBeNull()
  })

  it('still charts normally once the bars load', async () => {
    vi.mocked(productsApi.bars).mockResolvedValue({ symbol: 'SA', bars: [] })
    renderWorkspace(<ChartWorkspace />, { path: '/?symbol=SA' })

    await waitFor(() => expect(screen.getAllByTestId('chart').length).toBeGreaterThan(0))
    expect(screen.queryByText('No data downloaded for this product yet')).toBeNull()
  })
})
