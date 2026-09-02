# Backtesting — `runner.py`

Runs one strategy over a date range and writes a trade log, five charts and a
metrics summary.

```bash
python runner.py --symbols SA FG CF --start 2020-01-01 --end 2026-12-31 \
    --strategy double_ma --cash 100000
```

## Flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--symbols` | Products to load (weighted + contract data) | `SA FG CF BU RB HC C JM V` |
| `--start` / `--end` | Backtest window, `YYYY-MM-DD` | `2020-01-01` / `2026-12-31` |
| `--cash` | Initial equity (CNY) | `100000` |
| `--strategy` | Strategy short name, see `--help` for the discovered list | `double_ma` |
| `--slippage` | Fill slippage in price points, applied against the order | `0.0` |
| `--lots` | Lots per trade, for strategies that expose it | `1` |
| `--params-from` | Load params from an `optimize` `*_best.json` report | — |
| `--param NAME=VALUE` | Override one param; repeatable; beats `--params-from` | — |
| `--meta-model` | Path to a `meta_runner.py fit` artifact; gates entries | — |
| `--update-data` | Refresh exchange data before running | off |
| `--results-dir` | Output directory | `results/` |
| `--keep-last N` | Delete result files from all but the N most recent runs | off |
| `--quiet` / `--verbose` | Log level | info |

`--meta-model` is in-sample wherever the model was trained; use
[`meta_runner.py walkforward`](meta-labeling.md) to actually measure a filter.

## Execution model

The day loop has four phases: **OPEN → INTRABAR → SIGNAL → SETTLE**.

| Phase | What happens |
| --- | --- |
| OPEN | Rolls to the new calendar contract if the map changed, fills orders queued yesterday at today's open (± slippage), arms protective stops |
| INTRABAR | Stop-outs, checked against today's high/low; a gapped-through stop fills at the open, not the stop price |
| SIGNAL | `on_bar` runs; orders are queued, never filled on the bar that produced them |
| SETTLE | Mark to market on settlement prices; if available margin goes negative, all positions are force-liquidated that day |

Signals are computed on the OI-weighted series; fills happen on the calendar
main contract for that date. A strategy never names a physical contract, and
never handles a roll itself.

Costs come from `datafeed/products.py` per product: multiplier, margin ratio,
and either `commission_rate` (fraction of notional) or `commission_per_lot`.
Long and short are accounted symmetrically.

## Outputs

Written to `--results-dir`, timestamped per run:

| File | Content |
| --- | --- |
| `<Strategy>_trades_<ts>.csv` | One row per closed logical trade |
| `<Strategy>_equity_<ts>.png` | Equity curve |
| `<Strategy>_returns_<ts>.png` | Return curve |
| `<Strategy>_position_<ts>.png` | Position over time |
| `<Strategy>_signals_<ts>.png` | Price with entry/exit markers |
| `<Strategy>_summary_<ts>.png` | Metrics summary panel |

Trade log columns: `trade_id`, `open_date`, `close_date`, `direction`,
`symbol`, `contract`, `contracts`, `n_rolls`, `open_price`, `close_price`,
`size`, `gross_pnl`, `commission`, `net_pnl`, `margin_used`, `open_at_end`,
`forced`, `open_bar`, `close_bar`.

## Metrics

Printed at the end of a run and rendered into the summary chart:

- **Returns** — total, annualized, annualized volatility
- **Risk-adjusted** — Sharpe, Sortino, Calmar (risk-free 3% by default)
- **Drawdown** — max drawdown and days to recover it
- **Trades** — count, win rate, profit/loss ratio, profit factor, expectancy,
  average win / loss / holding days
- **Capital** — turnover, capital exposure, forced-liquidation count
- **Per symbol** — the same trade statistics broken out by product

Every run also reconciles `sum(net_pnl)` against the equity change and warns
if they drift, so an equity curve can never quietly pick up a term the trade
log does not explain.
