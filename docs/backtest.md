# Backtesting — `runner.py`

Runs one strategy over a date range and writes a trade log, five charts and a
metrics summary.

```bash
python runner.py --symbols SA CF RB --start 2020-01-01 --end 2026-12-31 \
    --strategy double_ma --cash 100000
```

## Flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--symbols` | Products to load (weighted + contract data) | `SA FG CF C` |
| `--start` / `--end` | Backtest window, `YYYY-MM-DD` | `2020-01-01` / `2026-12-31` |
| `--cash` | Initial equity (CNY) | `100000` |
| `--strategy` | Strategy short name, see `--help` for the discovered list | `double_ma` |
| `--slippage` | Fill slippage in ticks, applied against the order | `0.0` |
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
| OPEN | Rolls to the new calendar contract if the map changed, fills orders queued yesterday at today's open (± slippage) **subject to the margin check below**, arms the protective bracket (stop and take-profit) |
| INTRABAR | Bracket exits, checked against today's high/low; a level the open already gapped past fills at the open, not at the level. When one bar touches both, the **stop** wins — a daily bar records no path, and this is the reading that cannot flatter the result |
| SIGNAL | `on_bar` runs; orders are queued, never filled on the bar that produced them |
| SETTLE | Mark to market on settlement prices; if available margin goes negative, all positions are force-liquidated that day; if equity is still ≤ 0 after that, the account is blown up and the run stops |

Signals are computed on the OI-weighted series; fills happen on the calendar
main contract for that date. A strategy never names a physical contract, and
never handles a roll itself.

### Rolls

**A product holds exactly one contract at a time.** Everything keyed by
product — the protective bracket, the ledger's logical trade, the margin
model — assumes it, so the roll is what has to keep it true.

When the calendar moves, the whole position is closed on the leg it is held on
and reopened on the new one, at each contract's own open. If the old leg did
not print that day there is no exit price, and what happens next depends on
whether one is still coming:

- **Still trading** — an ordinary dark day. The roll waits, and orders queued
  meanwhile stay on the old leg rather than opening a second one on the
  calendar contract. With the old leg dark they simply defer, exactly as they
  would on any other dark bar.
- **Finished printing** — expired, or delisted mid-run. Waiting is waiting
  forever, so the leg is closed at its carried mark and the exposure moves on.
  That fill is at a price nobody traded, so the run warns about it and counts
  it in `stranded_rolls`. A non-zero count usually means the product's
  `main_months` no longer match where the liquidity is, and the calendar is
  holding contracts to the end of their life — check them against the cached
  open interest.

**The bracket is re-anchored on the new contract's price scale.** Two calendar
contracts run a basis of a few percent, so a level carried across a roll
unchanged is a level being compared against the wrong prices — on a
backwardated roll that puts a long's stop *above* the market, and the INTRABAR
gap rule then flattens the position at that same open. A `distance` needs no
arithmetic: it is re-resolved against the new leg's `avg_entry`, so the basis
cancels. An explicit `price` is shifted by the basis the roll realized, spec
included, so a position that later closes and reopens does not re-arm on the
stale scale. This is the reason strategies are pointed at distances.

Costs come from `datafeed/products.py` per product: multiplier, margin ratio,
and either `commission_rate` (fraction of notional) or `commission_per_lot`.
Long and short are accounted symmetrically.

### Capital limits

Two guards keep a run from trading capital it does not have.

**Orders are checked before they fill.** At OPEN, an order that would raise the
margin requirement past what equity covers is **rejected** — not filled, not
deferred, and not retried. The strategy is free to signal the same position
again next bar; nothing replays it on its behalf, because a replayed order
would open at a price no signal chose. Orders that do *not* raise the
requirement — closes, reductions, flips into a cheaper leg — always pass, so a
rejection can never trap a position inside a margin call. Rolls and forced
liquidations skip the check entirely: a roll is the same exposure on a
different contract, and a liquidation is the remedy for insolvency.

The run reports how many orders it refused. A non-zero count means the metrics
describe a **smaller book than the strategy asked for** — raise `--cash` or cut
the universe before reading them as the strategy's performance.

**A blown-up account stops the run.** If equity is still ≤ 0 at SETTLE after
the forced liquidation has had its chance, the remaining bars are not traded:
every one of them would be a position opened on capital that no longer exists.
The summary says so above the numbers, and the metrics then cover only the
truncated window. (Equity can land *below* zero on the way out — a gap through
the margin is exactly how a real account goes negative — but it cannot keep
trading afterwards.)

`--slippage` is quoted in **ticks**, not price points: each fill is moved
against the order by `slippage × tick_size`, with `tick_size` read per product
from the same registry. One setting therefore means the same thing across the
universe — `--slippage 1` is one minimum price increment on gold (0.02) and on
copper (10) alike.

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
- **Capital** — turnover, capital exposure, forced-liquidation count, rejected-order count, blow-up flag
- **Per symbol** — the same trade statistics broken out by product

Every run also reconciles `sum(net_pnl)` against the equity change and warns
if they drift, so an equity curve can never quietly pick up a term the trade
log does not explain.
