import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { productsApi, runsApi, strategiesApi } from '../api/endpoints'
import { PRODUCTS, STRATEGIES, renderWorkspace, runDetail, runRow } from '../test/utils'
import { ChartWorkspace } from './ChartWorkspace'

vi.mock('../api/endpoints', () => ({
  productsApi: { list: vi.fn(), bars: vi.fn(), roll: vi.fn(), get: vi.fn(), exchanges: vi.fn() },
  dataApi: { update: vi.fn() },
  strategiesApi: { list: vi.fn() },
  backtestApi: { start: vi.fn() },
  runsApi: { list: vi.fn(), get: vi.fn(), price: vi.fn(), remove: vi.fn() },
  jobsApi: { get: vi.fn(), cancel: vi.fn(), streamUrl: (id: string) => `/api/jobs/${id}/stream` },
}))

// Canvas-backed and irrelevant to the logic under test.
vi.mock('../components/EChart', () => ({ EChart: () => <div data-testid="chart" /> }))

beforeEach(() => {
  vi.mocked(productsApi.list).mockResolvedValue(PRODUCTS)
  vi.mocked(productsApi.bars).mockResolvedValue({ symbol: 'SA', bars: [] })
  vi.mocked(productsApi.roll).mockResolvedValue({ symbol: 'SA', roll: [] })
  vi.mocked(strategiesApi.list).mockResolvedValue(STRATEGIES)
  vi.mocked(runsApi.list).mockResolvedValue([runRow('run-a', ['SA', 'CF']), runRow('run-b', ['AG'])])
  vi.mocked(runsApi.get).mockImplementation(async (id: string) =>
    id === 'run-a' ? runDetail('run-a', ['SA', 'CF']) : runDetail('run-b', ['AG']),
  )
  vi.mocked(runsApi.price).mockResolvedValue({
    symbol: 'X', dates: [], open: [], high: [], low: [], close: [], volume: [], oi: [], signals: [],
  } as never)
})

describe('opening a past run from the workspace history tab', () => {
  it("re-charts to the opened run's first symbol and never mixes the old symbol with the new run in its price fetch", async () => {
    // Opening a run prefills its universe too (see HistoryPanel/
    // WorkspaceContext.prefillBacktest), so the chart follows the run
    // rather than staying on whatever was charted before -- the price call
    // that fires must pair the *new* run with the *new* symbol, never the
    // old symbol with the new run id or vice versa.
    const user = userEvent.setup()
    renderWorkspace(<ChartWorkspace />, { path: '/?symbol=SA&tab=runs' })

    // Row 0 is the header; the two runs follow in the order the API listed them.
    const [, rowA] = await screen.findAllByRole('row')

    await user.click(rowA)
    await waitFor(() => expect(runsApi.price).toHaveBeenCalledWith('run-a', 'SA'))

    vi.mocked(runsApi.price).mockClear()
    // Opening a row also prefills the Backtest tab and switches to it, so
    // reaching the next row means going back to History first -- the same
    // thing a user comparing two runs would do.
    await user.click(screen.getByRole('button', { name: 'Backtest History' }))
    const [, , freshRowB] = await screen.findAllByRole('row')
    await user.click(freshRowB)

    // run-b's own symbols are ['AG'] -- opening it re-charts to AG (its
    // first symbol) instead of leaving the chart on SA, which run-b never
    // traded and would otherwise have 404'd the price fetch.
    await waitFor(() => expect(runsApi.price).toHaveBeenCalledWith('run-b', 'AG'))
    expect(runsApi.price).not.toHaveBeenCalledWith('run-b', 'SA')
    expect(runsApi.price).not.toHaveBeenCalledWith('run-a', 'AG')

    // The base series never depended on the run's price call -- it always
    // comes from /products/{code}/bars, so the chart is still on screen.
    expect(screen.getAllByTestId('chart').length).toBeGreaterThan(0)
  })
})

/** A row's `.code` and `.name` cells both read e.g. "SA" in the test
 * fixtures (PRODUCTS derives `name` from `code`), so `getByText` alone is
 * ambiguous within one row. Find the row by its `.code` cell specifically. */
async function findProductRow(list: HTMLElement, code: string) {
  return waitFor(() => {
    const row = Array.from(list.querySelectorAll('.product-row')).find(
      (r) => r.querySelector('.code')?.textContent === code,
    )
    if (!row) throw new Error(`no product row for ${code}`)
    return row as HTMLElement
  })
}

describe('the product sidebar', () => {
  it('single-clicking a product toggles it into and out of the backtest universe used by the Backtest tab', async () => {
    const user = userEvent.setup()
    renderWorkspace(<ChartWorkspace />, { path: '/?symbol=SA&tab=backtest' })

    const list = await screen.findByRole('listbox')
    const cfRow = await findProductRow(list, 'CF')
    expect(cfRow).toHaveAttribute('aria-selected', 'false')

    await user.click(cfRow)
    expect(cfRow).toHaveAttribute('aria-selected', 'true')

    // The Backtest tab's universe chips (in .panel-rail) pick up the newly
    // toggled product alongside the sidebar's own row for it.
    const rail = document.querySelector('.panel-rail')!
    await waitFor(() => expect(within(rail as HTMLElement).getByText('CF')).toBeInTheDocument())

    await user.click(cfRow)
    expect(cfRow).toHaveAttribute('aria-selected', 'false')
  })

  it('double-clicking a product switches the main chart to it; single-clicking the charted row is a no-op on the universe', async () => {
    const user = userEvent.setup()
    renderWorkspace(<ChartWorkspace />, { path: '/?symbol=SA' })

    const list = await screen.findByRole('listbox')
    const saRow = await findProductRow(list, 'SA')
    expect(saRow.className).toContain('is-charted')

    // The charted product is structurally always in the universe, so a
    // single click on its own row can't toggle anything.
    await user.click(saRow)
    expect(saRow).toHaveAttribute('aria-selected', 'true')

    const cfRow = await findProductRow(list, 'CF')
    await user.dblClick(cfRow)
    expect(cfRow.className).toContain('is-charted')
    expect(saRow.className).not.toContain('is-charted')
  })
})
