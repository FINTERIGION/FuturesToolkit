// Validated categorical/status palette from the dataviz skill
// (references/palette.md). Kept as plain hex arrays because ECharts renders
// to canvas and needs literal colors, not CSS custom properties -- see
// useChartTheme() for how these pair with light/dark chart surfaces.

export const CATEGORICAL_LIGHT = [
  '#2a78d6', // 1 blue
  '#eb6834', // 2 orange
  '#1baf7a', // 3 aqua
  '#eda100', // 4 yellow
  '#e87ba4', // 5 magenta
  '#008300', // 6 green
  '#4a3aa7', // 7 violet
  '#e34948', // 8 red
]

export const CATEGORICAL_DARK = [
  '#3987e5',
  '#d95926',
  '#199e70',
  '#c98500',
  '#d55181',
  '#008300',
  '#9085e9',
  '#e66767',
]

export const STATUS = {
  good: '#0ca30c',
  warning: '#fab219',
  serious: '#ec835a',
  critical: '#d03b3b',
}

// Candlestick up/down is a Chinese-market financial convention, not the
// skill's generic blue/red diverging pair: red = price rose (阳线), green =
// price fell (阴线) -- the reverse of the US convention this toolkit's
// audience would otherwise expect.
export const CANDLE = {
  up: '#e34948', // red: price rose
  down: '#1baf7a', // green: price fell (reusing the aqua/green categorical slot)
}

export const CHART_SURFACE = { light: '#fcfcfb', dark: '#1a1a19' }
export const INK = {
  light: { primary: '#0b0b0b', secondary: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7' },
  dark: { primary: '#ffffff', secondary: '#c3c2b7', muted: '#898781', grid: '#2c2c2a', axis: '#383835' },
}
