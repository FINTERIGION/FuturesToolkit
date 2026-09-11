import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { MockEventSource } from '../test/setup'
import { JobsProvider, useJobSlot } from './JobsProvider'

vi.mock('../api/endpoints', () => ({
  jobsApi: {
    get: vi.fn(),
    cancel: vi.fn(),
    streamUrl: (id: string) => `/api/jobs/${id}/stream`,
  },
}))

function BacktestConsumer() {
  const job = useJobSlot('backtest')
  return (
    <div>
      <button onClick={() => job.start('j1')}>start</button>
      <span data-testid="status">{job.state?.status ?? 'none'}</span>
    </div>
  )
}

/** Stands in for the bottom drawer: mounting/unmounting `BacktestConsumer`
 * is what a real tab switch does to `BacktestPanel`, while `JobsProvider`
 * itself -- like the real one, mounted once at the workspace's root --
 * never unmounts. */
function Harness() {
  const [showBacktestTab, setShowBacktestTab] = useState(true)
  return (
    <JobsProvider>
      <button onClick={() => setShowBacktestTab((v) => !v)}>switch tab</button>
      {showBacktestTab && <BacktestConsumer />}
    </JobsProvider>
  )
}

describe('JobsProvider', () => {
  it('keeps a job streaming across the consuming panel unmounting and remounting', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    await user.click(screen.getByRole('button', { name: 'start' }))
    expect(MockEventSource.instances).toHaveLength(1)

    MockEventSource.instances[0].emit({ type: 'state', status: 'running', progress: 0.2, message: 'Running', progress_data: [] })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('running'))

    // Switch away from the backtest tab (unmount) and back (remount) --
    // the drawer does exactly this on every tab change.
    await user.click(screen.getByRole('button', { name: 'switch tab' }))
    expect(screen.queryByTestId('status')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'switch tab' }))

    // The remounted panel picks the run straight back up from context --
    // no reconnect, and the status it lost while unmounted is not lost.
    expect(MockEventSource.instances).toHaveLength(1)
    expect(screen.getByTestId('status')).toHaveTextContent('running')

    // A frame that arrives while the panel is live again still reaches it,
    // over the same connection.
    MockEventSource.instances[0].emit({ type: 'state', status: 'done', progress: 1, message: 'Done', progress_data: [] })
    await waitFor(() => expect(screen.getByTestId('status')).toHaveTextContent('done'))
    expect(MockEventSource.instances).toHaveLength(1)
  })
})
