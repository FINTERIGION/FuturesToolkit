import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import { jobsApi } from '../api/endpoints'
import { MockEventSource } from '../test/setup'
import { useJob } from './useJob'

vi.mock('../api/endpoints', () => ({
  jobsApi: {
    get: vi.fn(),
    cancel: vi.fn(),
    streamUrl: (id: string) => `/api/jobs/${id}/stream`,
  },
}))

const mockGet = vi.mocked(jobsApi.get)
const mockCancel = vi.mocked(jobsApi.cancel)

/** Start a job and drop its event stream, which is what puts the hook into the
 * polling path every test here is about. */
function startAndDropTheStream(result: { current: ReturnType<typeof useJob> }) {
  act(() => result.current.start('job-1'))
  const source = MockEventSource.instances.at(-1)!
  act(() => source.emit({ type: 'state', status: 'running', progress: 0.5, message: 'working' }))
  act(() => source.fail())
  return source
}

describe('useJob polling after the stream drops', () => {
  beforeEach(() => {
    mockGet.mockReset()
    mockCancel.mockReset()
  })

  it('stops polling and reports the job lost when the server 404s', async () => {
    // The job registry is an in-memory dict, so a backend restart loses it for
    // good. Retrying that forever left the panel on "running" with the Run
    // button disabled and a request going out every 3s for the life of the tab.
    mockGet.mockRejectedValue(new ApiError(404, 'Unknown job'))

    const { result } = renderHook(() => useJob())
    startAndDropTheStream(result)

    await waitFor(() => expect(result.current.lost).toBe(true))

    const callsWhenLost = mockGet.mock.calls.length
    await new Promise((r) => setTimeout(r, 60))
    expect(mockGet.mock.calls.length).toBe(callsWhenLost)

    // And the page can start another run: a job nobody can ask about is not
    // one the UI should keep waiting on.
    expect(result.current.isActive).toBe(false)
  })

  it('keeps polling through a network error, which may just be a restart', async () => {
    mockGet.mockRejectedValue(new TypeError('Failed to fetch'))

    const { result } = renderHook(() => useJob())
    startAndDropTheStream(result)

    await waitFor(() => expect(mockGet).toHaveBeenCalled())
    expect(result.current.lost).toBe(false)
  })

  it('resets lost when a new job is started', async () => {
    mockGet.mockRejectedValue(new ApiError(404, 'Unknown job'))
    const { result } = renderHook(() => useJob())
    startAndDropTheStream(result)
    await waitFor(() => expect(result.current.lost).toBe(true))

    act(() => result.current.start('job-2'))
    expect(result.current.lost).toBe(false)
  })
})

describe('useJob cancel', () => {
  beforeEach(() => {
    mockGet.mockReset()
    mockCancel.mockReset()
  })

  it('re-syncs state instead of leaving a rejected cancel unhandled', async () => {
    // Clicking Cancel just as the job finishes gets a 409 back. That rejection
    // used to go nowhere: the button did nothing and the panel went on showing
    // a job it had already been told was over.
    const { result } = renderHook(() => useJob())
    act(() => result.current.start('job-1'))
    const source = MockEventSource.instances.at(-1)!
    act(() => source.emit({ type: 'state', status: 'running', progress: 0.9, message: 'nearly' }))

    mockCancel.mockRejectedValue(new ApiError(409, "Job 'job-1' is already done"))
    mockGet.mockResolvedValue({ status: 'done', progress: 1, message: 'finished' } as never)

    act(() => result.current.cancel())

    await waitFor(() => expect(result.current.state?.status).toBe('done'))
    expect(result.current.isActive).toBe(false)
  })
})
