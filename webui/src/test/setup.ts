import '@testing-library/jest-dom/vitest'
// Initialises the shared i18next instance as a side effect, the same way
// main.tsx does, so `useTranslation` resolves real strings in tests rather
// than echoing key paths back.
import '../i18n'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'

// jsdom implements neither of these, and both are read during the first render
// of pages under test: `useIsDarkMode` calls `matchMedia`, and `useJob` opens
// an `EventSource` as soon as a job starts.
class MockEventSource {
  static instances: MockEventSource[] = []
  onopen: (() => void) | null = null
  onmessage: ((e: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  closed = false

  // A plain field assignment, not a `public url` parameter property: this
  // project compiles with `erasableSyntaxOnly`, which rejects the shorthand.
  url: string

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  close() {
    this.closed = true
  }

  /** Drive the stream from a test the way the server would. */
  emit(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }

  fail() {
    this.onerror?.()
  }
}

beforeEach(() => {
  MockEventSource.instances.length = 0

  vi.stubGlobal('EventSource', MockEventSource)
  vi.stubGlobal(
    'matchMedia',
    (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
      onchange: null,
    }),
  )
  // jsdom implements no layout, so this is a no-op stub rather than a mock:
  // BacktestPage scrolls the results panel into view when a history row is
  // opened, and the missing method would throw mid-click.
  Element.prototype.scrollIntoView = () => {}

  localStorage.clear()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

export { MockEventSource }
