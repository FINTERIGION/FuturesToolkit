import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { backtestApi, runsApi, strategiesApi } from '../api/endpoints'
import { STRATEGIES, renderWorkspace } from '../test/utils'
import type { BacktestFieldsPrefill } from './BacktestPanel'
import { BacktestPanel } from './BacktestPanel'

vi.mock('../api/endpoints', () => ({
  strategiesApi: { list: vi.fn() },
  backtestApi: { start: vi.fn() },
  runsApi: { list: vi.fn(), get: vi.fn(), price: vi.fn(), remove: vi.fn() },
  jobsApi: { get: vi.fn(), cancel: vi.fn(), streamUrl: (id: string) => `/api/jobs/${id}/stream` },
}))

/** The param inputs carry no htmlFor, so reach them through the field the
 * param name labels. */
function paramInput(name: string) {
  const field = screen.getByText(name).closest('.field')!
  return within(field as HTMLElement).getByRole('spinbutton')
}

beforeEach(() => {
  vi.mocked(strategiesApi.list).mockResolvedValue(STRATEGIES)
  vi.mocked(backtestApi.start).mockResolvedValue({ job_id: 'j1', run_id: 'r1' })
  vi.mocked(runsApi.get).mockResolvedValue({
    id: 'r1', kind: 'backtest', created_at: 0, strategy: null, symbols: [], start: null, end: null,
    cash: null, slippage: null, status: 'done', params: {}, metrics: null, error: null,
    equity_records: [], trade_logs: [], deferred: {}, symbols_with_price: [],
  })
})

function renderPanel(prefill?: BacktestFieldsPrefill) {
  return renderWorkspace(<BacktestPanel prefill={prefill} />, { path: '/?symbol=SA' })
}

describe('prefill from a past run', () => {
  it('keeps the prefilled params when the remembered strategy is a different one', async () => {
    // The ordinary case: the sticky strategy is whatever was run last, and the
    // run being reopened used something else. Applying the prefill used
    // to change `strategyKey` after the first render, which re-ran the
    // params-reset effect and wiped exactly the params the user asked for --
    // leaving a form that looked filled in but held the strategy's defaults.
    localStorage.setItem('ft.strategy', JSON.stringify('double_ma'))

    renderPanel({ strategy: 'chan_theory', start: '2020-01-01', end: '2024-01-01', params: { atr_mult: 3.7 } })

    await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('chan_theory'))
    expect(paramInput('atr_mult')).toHaveValue(3.7)
  })

  it('keeps them when the remembered strategy happens to match too', async () => {
    localStorage.setItem('ft.strategy', JSON.stringify('chan_theory'))

    renderPanel({ strategy: 'chan_theory', params: { atr_mult: 4.25 } })

    await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('chan_theory'))
    expect(paramInput('atr_mult')).toHaveValue(4.25)
  })

  it('carries the cost assumptions the run was made under', async () => {
    // Without these the rerun reproduces nothing: the panel's own defaults are
    // 200000 / 0, and a run made at 100000 / 1.5 is a different backtest.
    localStorage.setItem('ft.cash', JSON.stringify(200000))
    localStorage.setItem('ft.slippage', JSON.stringify(0))

    renderPanel({ strategy: 'chan_theory', cash: 100000, slippage: 1.5 })

    await waitFor(() => expect(screen.getByRole('combobox')).toHaveValue('chan_theory'))
    const cash = screen.getByText('Cash').closest('.field')!
    const slippage = screen.getByText('Slippage').closest('.field')!
    expect(within(cash as HTMLElement).getByRole('spinbutton')).toHaveValue(100000)
    expect(within(slippage as HTMLElement).getByRole('spinbutton')).toHaveValue(1.5)
  })

  it('still clears params when the user switches strategy themselves', async () => {
    const user = userEvent.setup()
    localStorage.setItem('ft.strategy', JSON.stringify('chan_theory'))
    renderPanel({ strategy: 'chan_theory', params: { atr_mult: 4.25 } })

    await waitFor(() => expect(paramInput('atr_mult')).toHaveValue(4.25))
    await user.selectOptions(screen.getByRole('combobox'), 'double_ma')

    await waitFor(() => expect(paramInput('fast')).toHaveValue(10))
    expect(paramInput('slow')).toHaveValue(30)
  })
})

describe('slippage validation', () => {
  it('blocks the run and explains why when slippage is negative', async () => {
    localStorage.setItem('ft.strategy', JSON.stringify('double_ma'))
    renderPanel()

    const runButton = await screen.findByRole('button', { name: /run backtest/i })
    await waitFor(() => expect(runButton).toBeEnabled())

    const slippage = screen.getByText('Slippage').closest('.field')!
    const input = within(slippage as HTMLElement).getByRole('spinbutton')
    // `fireEvent.change` rather than `user.type`: jsdom discards a number
    // input's value while it is transiently invalid, so typing "-5" a
    // character at a time lands on 5, not -5. A real browser keeps the minus.
    fireEvent.change(input, { target: { value: '-5' } })

    await waitFor(() => expect(runButton).toBeDisabled())
    expect(screen.getByText(/cannot be negative/i)).toBeInTheDocument()
    expect(backtestApi.start).not.toHaveBeenCalled()
  })
})
