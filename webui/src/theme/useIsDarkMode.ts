import { useEffect, useState } from 'react'

/** Tracks the OS `prefers-color-scheme`, which is the only dark-mode signal
 * this app has -- there is no manual light/dark toggle in scope, only the
 * language switch. The CSS side already handles this via a media query
 * (see index.css); this hook exists so ECharts option-builders, which need
 * literal hex colors rather than CSS custom properties, can react to the
 * same signal. */
export function useIsDarkMode(): boolean {
  const [dark, setDark] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches,
  )

  useEffect(() => {
    const mql = window.matchMedia('(prefers-color-scheme: dark)')
    const listener = (e: MediaQueryListEvent) => setDark(e.matches)
    mql.addEventListener('change', listener)
    return () => mql.removeEventListener('change', listener)
  }, [])

  return dark
}
