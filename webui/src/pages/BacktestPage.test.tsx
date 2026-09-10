import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { backtestApi, dataApi, productsApi, runsApi, strategiesApi } from '../api/endpoints'
import { COVERAGE, PRODUCTS, STRATEGIES, renderPage } from '../test/utils'
import { BacktestPage } from './BacktestPage'

vi.mock('../api/endpoints', () => ({
  strategiesApi: { list: vi.fn() },
  productsApi: { list: vi.fn() },
  dataApi: { coverage: vi.fn() },
  backtestApi: { start: vi.fn() },
  runsApi: { list: vi.fn(), get: vi.fn(), price: vi.fn(), remove: vi.fn() },
  jobsApi: { get: vi.fn(), cancel: vi.fn(), streamUrl: (id: string) => `/api/jobs/${id}/stream` },
}))

// Canvas-backed and irrelevant to the logic under test here.
vi.mock('../components/EChart', () => ({ EChart: () => <div data-testid="chart" /> }))

/** The param inputs carry no htmlFor, so reach them through the field the
 * param name labels. */
function paramInput(name: string) {
  const field = screen.getByText(name).closest('.field')!
  return within(field as HTMLElement).getByRole('spinbutton')
}

beforeEach(() => {
  vi.mocked(strategiesApi.list).mockResolvedValue(STRATEGIES)
  vi.mocked(productsApi.list).mockResolvedValue(PRODUCTS)
  vi.mocked(dataApi.coverage).mockResolvedValue(COVERAGE)
  vi.mocked(backtestApi.start).mockResolvedValue({ job_id: 'j1', run_id: 'r1' })
  vi.mocked(runsApi.list).mockResolvedValue([runRow('run-a', ['SA', 'CF']), runRow('run-b', ['AG'])])
  vi.mocked(runsApi.get).mockImplementation(async (id: string) => runDetail(id) as never)
  vi.mocked(runsApi.price).mockResolvedValue({
    symbol: 'X', dates: [], open: [], high: [], low: [], close: [], volume: [], oi: [], signals: [],
  } as never)
})

function runRow(id: string, symbols: string[]) {
  return {
    id, kind: 'backtest', created_at: 1_700_000_000, strategy: 'DoubleMaStrategy', symbols,
    start: '2020-01-01', end: '2024-01-01', cash: 200000, slippage: 0, status: 'done',
    params: {}, metrics: { sharpe_ratio: 1, blown_up: false }, error: null,
  }
}

function runDetail(id: string) {
  const symbols = id === 'run-a' ? ['SA', 'CF'] : ['AG']
  return {
    ...runRow(id, symbols),
    equity_records: [], trade_logs: [], deferred: {}, symbols_with_price: symbols,
  }
}

describe('opening a past run from the history table', () => {
  it("does not carry the previous run's symbol selection into the next one", async () => {
    // The results panel keeps the active tab and the per-symbol selection in
    // its own state. Without a key it is reused across runs, so opening a run
    // whose symbols differ went on asking for a symbol it does not have --
    // `/runs/run-b/price/CF` -- which 404s into a blank panel explaining
    // nothing.
    const user = userEvent.setup()
    localStorage.setItem('ft.strategy', JSON.stringify('double_ma'))
    renderPage(<BacktestPage />)

    // Row 0 is the header; the two runs follow in the order the API listed them.
    const [, rowA, rowB] = await screen.findAllByRole('row')

    await user.click(rowA)
    await user.click(await screen.findByRole('button', { name: 'Price & Signals' }))
    await user.click(await screen.findByRole('button', { name: 'CF' }))
    await waitFor(() => expect(runsApi.price).toHaveBeenCalledWith('run-a', 'CF'))

    vi.mocked(runsApi.price).mockClear()
    await user.click(rowB)

    await waitFor(() => expect(runsApi.get).toHaveBeenCalledWith('run-b'))
    await waitFor(() => expect(runsApi.price).toHaveBeenCalled())
    for (const [runId, symbol] of vi.mocked(runsApi.price).mock.calls) {
      expect({ runId, symbol }).toEqual({ runId: 'run-b', symbol: 'AG' })
    }
  })
})

describe('Send to Backtest prefill', () => {
  it('keeps the tuned params when the remembered strategy is a different one', async () => {
    // The ordinary case: the sticky strategy is whatever was run last, and the
    // report being sent over is for something else. Applying the prefill used
    // to change `strategyKey` after the first render, which re-ran the
    // params-reset effect and wiped exactly the params the user asked for --
    // leaving a form that looked filled in but held the strategy's defaults.
    localStorage.setItem('ft.strategy', JSON.stringify('double_ma'))

    renderPage(<BacktestPage />, {
      strategy: 'chan_theory',
      symbols: ['SA'],
      start: '2020-01-01',
      end: '2024-01-01',
      params: { atr_mult: 3.7 },
    })

    await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('chan_theory'))
    expect(paramInput('atr_mult')).toHaveValue(3.7)
  })

  it('keeps them when the remembered strategy happens to match too', async () => {
    localStorage.setItem('ft.strategy', JSON.stringify('chan_theory'))

    renderPage(<BacktestPage />, { strategy: 'chan_theory', params: { atr_mult: 4.25 } })

    await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('chan_theory'))
    expect(paramInput('atr_mult')).toHaveValue(4.25)
  })

  it('carries the cost assumptions the study was tuned under', async () => {
    // Without these the run reproduces nothing: the panel's own defaults are
    // 200000 / 0, and a report tuned at 100000 / 1.0 is a different backtest.
    localStorage.setItem('ft.cash', JSON.stringify(200000))
    localStorage.setItem('ft.slippage', JSON.stringify(0))

    renderPage(<BacktestPage />, { strategy: 'chan_theory', cash: 100000, slippage: 1.5 })

    await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('chan_theory'))
    const cash = screen.getByText('Cash').closest('.field')!
    const slippage = screen.getByText('Slippage').closest('.field')!
    expect(within(cash as HTMLElement).getByRole('spinbutton')).toHaveValue(100000)
    expect(within(slippage as HTMLElement).getByRole('spinbutton')).toHaveValue(1.5)
  })

  it('still clears params when the user switches strategy themselves', async () => {
    // The reset is not the bug -- losing it would leave one strategy's params
    // attached to another's run.
    const user = userEvent.setup()
    localStorage.setItem('ft.strategy', JSON.stringify('chan_theory'))
    renderPage(<BacktestPage />, { strategy: 'chan_theory', params: { atr_mult: 4.25 } })

    await waitFor(() => expect(paramInput('atr_mult')).toHaveValue(4.25))
    await user.selectOptions(screen.getByRole('combobox'), 'double_ma')

    await waitFor(() => expect(paramInput('fast')).toHaveValue(10))
    expect(paramInput('slow')).toHaveValue(30)
  })
})

describe('slippage validation', () => {
  it('blocks the run and explains why when slippage is negative', async () => {
    localStorage.setItem('ft.strategy', JSON.stringify('double_ma'))
    localStorage.setItem('ft.symbols', JSON.stringify(['SA']))
    renderPage(<BacktestPage />)

    const runButton = await screen.findByRole('button', { name: /run backtest/i })
    await waitFor(() => expect(runButton).toBeEnabled())

    const slippage = screen.getByText('Slippage').closest('.field')!
    const input = within(slippage as HTMLElement).getByRole('spinbutton')
    // `fireEvent.change` rather than `user.type`: jsdom discards a number
    // input's value while it is transiently invalid, so typing "-5" a
    // character at a time lands on 5, not -5. A real browser keeps the minus.
    // What matters here is the state the field ends up in, so set it directly.
    fireEvent.change(input, { target: { value: '-5' } })

    await waitFor(() => expect(runButton).toBeDisabled())
    expect(screen.getByText(/cannot be negative/i)).toBeInTheDocument()
    expect(backtestApi.start).not.toHaveBeenCalled()
  })
})
