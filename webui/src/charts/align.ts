/** The nearest date actually present in `axis` (sorted ascending ISO
 * strings), searching forward (`ceil`) or backward (`floor`) from `iso`.
 * A category-axis `markArea`/`markLine` renders nothing for a value that
 * is not literally in `xAxis.data` -- and a backtest's start/end date is
 * routinely a weekend or a holiday with no bar at all. Returns `null` when
 * the search runs off either end (the window starts after the last bar, or
 * ends before the first one). */
export function nearestCategory(axis: string[], iso: string, dir: 'floor' | 'ceil'): string | null {
  if (axis.length === 0) return null
  if (dir === 'ceil') {
    for (const d of axis) if (d >= iso) return d
    return null
  }
  for (let i = axis.length - 1; i >= 0; i--) if (axis[i] <= iso) return axis[i]
  return null
}
