import { useEffect, useState } from 'react'

const PREFIX = 'ft.'

/**
 * `useState`, but the value survives a reload and is shared by every page
 * that passes the same `key`.
 *
 * Backtest and Optimize ask for the same universe, date range and cash, and
 * retyping them on every visit was the single most repetitive thing in the
 * panel. Persisting them under one key each means picking symbols on either
 * page carries over to the other.
 *
 * Every access is wrapped: `localStorage` throws outright in a few real
 * contexts (Safari private mode, a browser set to block site data), and a
 * value written by an older build can fail to parse. Either way the caller
 * gets `initial` and the page renders -- a remembered form field is never
 * worth a blank screen.
 */
export function useStickyState<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(PREFIX + key)
      return raw === null ? initial : (JSON.parse(raw) as T)
    } catch {
      return initial
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify(value))
    } catch {
      // Storage unavailable or full: the field still works for this session.
    }
  }, [key, value])

  return [value, setValue]
}
