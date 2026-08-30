# FuturesToolkit

A self-built daily-bar backtesting engine for **Zhengzhou Commodity Exchange** futures — **SA (soda ash)**, **FG (glass)**, and **CF (cotton)**. It downloads historical data from the exchange, builds open-interest–weighted daily bars for **signals**, executes on calendar contracts (January / May / September), and exports equity curves, trade logs, and signal charts. The engine is self-built: a four-phase day loop (OPEN → INTRABAR → SIGNAL → SETTLE), a weighted-average-cost broker with multi-position margin accounting, a fill-driven trade ledger, and [TA-Lib](https://ta-lib.org/)-backed indicators.

## Features

- **Data pipeline** — fetch CZCE history per product, clean contract-level OHLC, and aggregate to OI-weighted continuous series
- **Calendar execution** — signals on the weighted series; fills on real 01/05/09 contracts, with automatic rolls at the open on the first live session of Apr / Aug / Dec
- **Multi-product** — load SA / FG / CF together; strategies loop over `ctx.symbols` and trade each independently
- **Futures cost model** — per-product multiplier, margin ratio, and commission from `datafeed/products.py`, with long/short-symmetric equity accounting and daily forced liquidation on a margin breach
- **Strategy API** — `Strategy` / `SetupContext` / `BarContext`, target-position order semantics (`ctx.set_target`), automatic indicator warmup skipping, protective stops
- **Metrics & reports** — Sharpe, Sortino, Calmar, max drawdown + recovery, win rate, turnover, capital exposure, per-symbol breakdown, forced-liquidation count
- **Charts** — equity, returns, position, price & signals, summary plots

## Requirements

- Python 3.11+

```
pandas==3.0.1
numpy==2.4.2
matplotlib==3.10.8
TA-Lib==0.7.1
pytest==9.1.1
```

## Installation

```bash
cd FuturesToolkit
pip install -r requirements.txt
```

## Quick Start

### 1. Download / refresh data

Data is pulled from CZCE and written to `data/{SA,FG,CF}.csv` and `data/{SA,FG,CF}_weighted.csv`. Historical years are cached under `cache/` and reused; only the current year is re-downloaded by default.

```bash
python -m datafeed.data_update              # incremental refresh (all products)
python -m datafeed.data_update FG CF        # selected products only
python -m datafeed.data_update --force      # re-download every year
python -m datafeed.data_update --rebuild-only
```

Or pass `--update-data` when running a backtest.

### 2. Run a backtest

```bash
python runner.py --symbols SA FG CF --start 2020-01-01 --end 2026-12-31 \
    --strategy double_ma --cash 100000
```

Run `python runner.py --help` for the full flag list (`--slippage`, `--lots`, `--keep-last N` to prune old result files, `--quiet` / `--verbose`). Outputs land in `results/` (charts, trade log CSV).

### 3. Switch strategies

Built-in examples, selected via `--strategy`:

```bash
python runner.py --strategy double_ma
python runner.py --strategy rsi_mean_reversion
python runner.py --strategy my_strategy      # template for your own strategy
```

## Project Layout

```
FuturesToolkit/
├── runner.py                  # CLI entry point
├── plotting.py                # Chart generation
├── core/                      # Engine internals
│   ├── types.py               #   Bar / Order / Fill / OrderType / Reason
│   ├── market.py              #   MarketData / ProductPanel / ContractSeries
│   ├── broker.py              #   cash, positions, margin, commission, forced liquidation
│   ├── ledger.py              #   fill-driven logical trade ledger
│   ├── engine.py              #   four-phase day loop, rolls, stops, deferral
│   ├── metrics.py             #   performance metrics
│   └── indicators.py          #   TA-Lib NaN guard
├── strategies/                # Strategy base + examples (tracked) + private modules (gitignored)
│   ├── base.py                #   Strategy / SetupContext / BarContext
│   ├── double_ma.py
│   ├── rsi_mean_reversion.py
│   └── my_strategy.py         #   template
├── datafeed/                  # Data pipeline
│   ├── data_update.py         #   CZCE download & OI-weighted aggregation
│   ├── data_manager.py        #   load / align / bundle data for the engine
│   ├── products.py            #   SA / FG / CF registry (multiplier, margin, commission)
│   └── roll_calendar.py       #   month → Jan/May/Sep contract map
├── tests/                     # pytest suite (synthetic MarketData fixtures)
├── data/                      # Generated CSVs (gitignored)
├── cache/                     # CZCE yearly raw files (gitignored)
└── results/                   # Backtest outputs (gitignored)
```

## Writing a Strategy

1. Subclass `Strategy` from `strategies.base`.
2. Precompute full-series indicators in `setup(ctx)` with `ctx.add_indicator` — the engine automatically skips `on_bar` until every registered indicator has a valid value, so strategy code never needs a NaN check.
3. Implement trading logic in `on_bar(ctx)`.
4. Add research strategies under `strategies/` (one class per file, gitignored beyond the tracked examples), or start from `strategies/my_strategy.py`. Register the class in `runner.py`'s `STRATEGIES` dict to select it via `--strategy`.

Minimal single-product sketch:

```python
import talib
from .base import Strategy

class MyStrategy(Strategy):
    params = {'period': 20, 'lots': 1}

    def setup(self, ctx):
        for sym in ctx.symbols:
            ctx.add_indicator('sma', sym, talib.SMA(ctx.close(sym), self.p['period']))

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            if not ctx.can_trade(sym):
                continue
            pos = ctx.position(sym)
            close = ctx.bar(sym).close
            if pos == 0 and close > ctx.ind('sma', sym):
                ctx.set_target(sym, self.p['lots'])
            elif pos > 0 and close < ctx.ind('sma', sym):
                ctx.close(sym)
```

`ctx.set_target(sym, lots)` is idempotent target-position semantics: it computes whatever delta is needed to reach `lots` and queues one order that fills at the next OPEN, reversing directly in a single fill rather than needing a separate close-then-reopen bar.

Available on `BarContext`:

| Accessor | Meaning |
|---|---|
| `ctx.bar(sym)` | `Bar(open, high, low, close, settle, volume, oi)` for the weighted series |
| `ctx.ind(name, sym)` | registered indicator value at this bar |
| `ctx.position(sym)` | net lots (signed) |
| `ctx.equity` / `cash` / `margin_used` / `available` | account state |
| `ctx.can_trade(sym)` | listed + real print today + mapped contract live today |
| `ctx.contract(sym)` | today's calendar contract code, e.g. `'SA509'` |
| `ctx.set_target(sym, lots)` / `buy(sym, lots)` / `sell(sym, lots)` / `close(sym)` | orders |
| `ctx.set_stop(sym, price=None, distance=None)` / `cancel_stop(sym)` | protective stop |
| `ctx.size_for_risk(sym, stop_distance, risk_pct)` | lots sized to risk `risk_pct` of equity on a stop-out |

Orders always fill against that day's calendar contract, resolved fresh by the engine at fill time — strategy code never has to think about which physical contract an order lands on, or handle rolls itself.

## Configuration Reference

| Flag | Meaning | Default |
|---|---|---|
| `--symbols` | Products whose weighted + contract data are loaded | `SA FG CF` |
| `--start` / `--end` | Backtest window | `2020-01-01` / `2026-12-31` |
| `--cash` | Starting equity (CNY) | `100000` |
| `--strategy` | `double_ma` / `rsi_mean_reversion` / `my_strategy` | `double_ma` |
| `--slippage` | Fill slippage in price points | `0.0` |
| `--lots` | Lots per trade (strategies that use it) | `1` |
| `--update-data` | Incremental refresh from CZCE before running | off |
| `--keep-last N` | Delete result files from all but the N most recent runs | off |

Multiplier, margin, and commission are **not** set on the CLI. Edit `datafeed/products.py` per product (SA/FG multiplier 20, CF 5). Commission is either `commission_rate` (fraction of notional) or `commission_per_lot` (fixed CNY per lot).

## Data Notes

- Source: [CZCE](http://www.czce.com.cn/) historical futures files for **SA**, **FG**, and **CF**.
- Contract-level history is cleaned and saved as `data/{symbol}.csv`.
- Daily continuous series uses open-interest weighting of **all listed contracts** that day → `data/{symbol}_weighted.csv`.
- Alignment onto the multi-product calendar **never looks ahead**. Days with no exchange print are marked `session=0`; open/high/low flatten to the last close for mark-to-market only. **No orders fill on those bars** (pending entries are deferred to the next real session). Each real contract stores only the bars where it actually printed — not padded to the full calendar.
- Execution, per product:
  - Dec–Mar trade the **May** contract, Apr–Jul the **September** contract, Aug–Nov the **next January** contract.
  - Roll on the first **live** session of April, August, and December: close the old contract at that day's **open**, open the new contract at its **open** (commission on both legs), folded into the same logical trade in the ledger.
  - A signal order queued the day before a roll fills against **that day's** calendar contract automatically — the engine resolves the target contract fresh at fill time, so there's no separate redirect step.
  - Protective stops (`ctx.set_stop`) are checked intrabar with `[low, high]`; a gap through the stop fills at the open instead of the stop price. They're parked on dark days and restored on the next print.
  - A margin breach at end-of-day settle force-liquidates every open position at that day's settle price.
- Trade CSV includes a `symbol` column and folds calendar rolls into their logical trade.
