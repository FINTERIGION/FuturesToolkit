# FuturesToolkit

A self-built daily-bar backtesting engine for **Zhengzhou Commodity Exchange** futures. It downloads historical data from the exchange, builds open-interest–weighted daily bars for **signals**, executes on each product's registered main contracts, and exports equity curves, trade logs, and signal charts.

## Features

- **Data pipeline** — fetch CZCE history per product, clean contract-level OHLC, and aggregate to OI-weighted continuous series
- **Calendar execution** — signals on the weighted series; fills on the real main contracts declared per product, rolled automatically at the open on the first live session of the month before delivery
- **Multi-product** — load SA / FG / CF / MA / TA / SR / OI together; strategies loop over `ctx.symbols` and trade each independently, or trade the universe as a single cross-section
- **Futures cost model** — per-product multiplier, margin ratio, and commission from `datafeed/products.py`, with long/short-symmetric equity accounting and daily forced liquidation on a margin breach
- **Strategy API** — `Strategy` / `SetupContext` / `BarContext`, target-position order semantics (`ctx.set_target`), automatic indicator warmup skipping, protective stops
- **Metrics & reports** — Sharpe, Sortino, Calmar, max drawdown & recovery, win rate, turnover, capital exposure, per-symbol breakdown, forced-liquidation count
- **Charts** — equity, returns, position, price & signals, summary plots
- **Research tools** — Optuna parameter optimization with anchored walk-forward validation, a locked holdout window, and overfitting diagnostics + meta-label signal filtering

## Requirements

- Python 3.11+

```
pandas==3.0.1
numpy==2.4.2
matplotlib==3.10.8
TA-Lib==0.7.1
pytest==9.1.1
optuna==4.9.0
scikit-learn==1.9.0
scipy==1.17.1
joblib==1.5.3
```

## Installation

```bash
cd FuturesToolkit
pip install -r requirements.txt
```

## Quick Start

### 1. Download / refresh data

Data is pulled from CZCE and written to `data/{SYMBOL}.csv` and `data/{SYMBOL}_weighted.csv` for each registered product. Historical years are cached under `cache/` and reused; only the current year is re-downloaded by default.

```bash
python -m datafeed.data_update              # incremental refresh (all products)
python -m datafeed.data_update SA CF        # selected products only
python -m datafeed.data_update --force      # re-download every year
python -m datafeed.data_update --rebuild-only
```

Or pass `--update-data` when running a backtest.

### 2. Run a backtest

```bash
python runner.py --symbols SA FG CF MA TA SR OI --start 2020-01-01 --end 2026-12-31 \
    --strategy double_ma --cash 100000
```

Run `python runner.py --help` for the full flag list (`--slippage`, `--lots`, `--keep-last N` to prune old result files, `--quiet` / `--verbose`). Outputs land in `results/` (charts, trade log CSV).

To run a tuned parameter set, point `--params-from` at an `optimize` report (see [Parameter Optimization](#parameter-optimization--meta-labeling)) — that gives you the full trade log and charts for those params, which `research_runner.py` itself does not produce.

```bash
python runner.py --strategy double_ma --params-from results/optuna/DoubleMaStrategy_<ts>_best.json
python runner.py --strategy double_ma --param fast_period=7 --param slow_period=40
```

### 3. Switch strategies

Built-in examples, selected via `--strategy`:

```bash
python runner.py --strategy double_ma
python runner.py --strategy rsi_mean_reversion
python runner.py --strategy cross_sectional_momentum
python runner.py --strategy my_strategy
```

## Project Layout

```
FuturesToolkit/
├── runner.py                  # CLI entry point
├── research_runner.py         # Research CLI: optimize / holdout / metalabel / meta-backtest
├── plotting.py                # Chart generation
├── core/                      # Engine internals
│   ├── types.py               #   Bar / Order / Fill / OrderType / Reason
│   ├── market.py              #   MarketData / ProductPanel / ContractSeries
│   ├── broker.py              #   cash, positions, margin, commission, forced liquidation
│   ├── ledger.py              #   fill-driven logical trade ledger
│   ├── engine.py              #   four-phase day loop, rolls, stops, deferral
│   ├── metrics.py             #   performance metrics
│   ├── indicators.py          #   TA-Lib NaN guard
│   └── params.py              #   Int / Float / Categorical search-space types
├── strategies/                # Strategy base + examples (tracked) + private modules (gitignored)
│   ├── base.py                #   Strategy / SetupContext / BarContext
│   ├── double_ma.py
│   ├── rsi_mean_reversion.py
│   ├── cross_sectional_momentum.py
│   └── my_strategy.py         #   template
├── research/                  # Optuna optimization + meta-labeling (strategy-agnostic)
│   ├── space.py               #   search-space resolution (declared / inferred / CLI override)
│   ├── splits.py              #   anchored walk-forward folds + locked holdout window
│   ├── warmup.py              #   exact per-strategy indicator warmup probing
│   ├── runner_api.py          #   single-window backtest with a leak-safe warmup pad
│   ├── objective.py           #   Optuna trial scoring
│   ├── optimize.py            #   study driver + holdout evaluation
│   ├── overfit.py             #   PBO (CSCV), Deflated Sharpe, IS/OOS decay, plateau check
│   ├── features.py            #   causal feature construction for meta-labeling
│   ├── gating.py              #   BarContext wrappers: event recording + P(win) gating
│   └── metalabel.py           #   event extraction, purged CV, fit, threshold, evaluation
├── datafeed/                  # Data pipeline
│   ├── data_update.py         #   CZCE download & OI-weighted aggregation
│   ├── data_manager.py        #   load / align / bundle data for the engine
│   ├── products.py            #   product registry (multiplier, margin, commission, roll months)
│   └── roll_calendar.py       #   date → main-month contract map (from products.py)
├── tests/                     # pytest suite (synthetic MarketData fixtures)
├── data/                      # Generated CSVs (gitignored)
├── cache/                     # CZCE yearly raw files (gitignored)
└── results/                   # Backtest outputs (gitignored), incl. results/optuna, results/meta
```

## Writing a Strategy

1. Subclass `Strategy` from `strategies.base`.
2. Precompute full-series indicators in `setup(ctx)` with `ctx.add_indicator` — the engine automatically skips `on_bar` until every registered indicator has a valid value, so strategy code never needs a NaN check.
3. Implement trading logic in `on_bar(ctx)`.
4. Add research strategies under `strategies/`, or start from `strategies/my_strategy.py`. Strategies are discovered automatically through `strategies.discover_strategies()`; the class name determines the `--strategy` value.
5. Optionally declare a `space` dict alongside `params` (using `Int(low, high)` / `Float(low, high)` / `Categorical(choices)` from `strategies.base`) to control how `research_runner.py optimize` searches this strategy's parameters. If omitted, a space is inferred from each param's default value.

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

There is no dedicated cross-sectional data structure — every product shares the same bar index and calendar, so `{sym: ctx.ind('mom', sym) for sym in ctx.symbols}` is the cross-section. See `strategies/cross_sectional_momentum.py` for a worked example that ranks the universe each rebalance, goes long the leaders and short the laggards, and budgets margin per leg rather than sizing each leg off the whole account.

Available on `BarContext`:


| Accessor                                                                          | Meaning                                                                   |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `ctx.bar(sym)`                                                                    | `Bar(open, high, low, close, settle, volume, oi)` for the weighted series |
| `ctx.ind(name, sym)`                                                              | registered indicator value at this bar                                    |
| `ctx.position(sym)`                                                               | net lots (signed)                                                         |
| `ctx.equity` / `cash` / `margin_used` / `available`                               | account state                                                             |
| `ctx.can_trade(sym)`                                                              | listed + real print today + mapped contract live today                    |
| `ctx.contract(sym)`                                                               | today's calendar contract code, e.g. `'SA509'`                            |
| `ctx.set_target(sym, lots)` / `buy(sym, lots)` / `sell(sym, lots)` / `close(sym)` | orders                                                                    |
| `ctx.set_stop(sym, price=None, distance=None)` / `cancel_stop(sym)`               | protective stop                                                           |
| `ctx.size_for_risk(sym, stop_distance, risk_pct)`                                 | lots sized to risk `risk_pct` of equity on a stop-out                     |

Orders always fill against that day's calendar contract, resolved fresh by the engine at fill time — strategy code never has to think about which physical contract an order lands on, or handle rolls itself.

## Configuration Reference

| Flag                | Meaning                                                 | Default                        |
| ------------------- | ------------------------------------------------------- | ------------------------------ |
| `--symbols`         | Products whose weighted + contract data are loaded      | `SA FG CF MA TA SR OI`         |
| `--start` / `--end` | Backtest window                                         | `2020-01-01` / `2026-12-31`    |
| `--cash`            | Starting equity (CNY)                                   | `100000`                       |
| `--strategy`        | `double_ma` / `rsi_mean_reversion` / `cross_sectional_momentum` / `my_strategy` | `double_ma` |
| `--slippage`        | Fill slippage in price points                           | `0.0`                          |
| `--lots`            | Lots per trade (strategies that use it)                 | `1`                             |
| `--update-data`     | Incremental refresh from CZCE before running            | off                             |
| `--keep-last N`     | Delete result files from all but the N most recent runs | off                             |

Multiplier, margin, and commission are **not** set on the CLI. Edit `datafeed/products.py` per product. Commission is either `commission_rate` (fraction of notional) or `commission_per_lot` (fixed CNY per lot). Rolling is registry-driven too: `main_months` lists the delivery months a product actually trades and `roll_lead_months` (default `1`) says how far ahead of delivery to leave them. Both are optional and default to the CZCE `(1, 5, 9)` cycle with a one-month lead.

## Parameter Optimization & Meta-Labeling

`research_runner.py` works on any strategy `strategies.discover_strategies()` finds — no strategy-specific code required. It has five subcommands:

```bash
# 1. See what will be tuned
python research_runner.py show-space --strategy double_ma

# 2. Optuna anchored walk-forward search
python research_runner.py optimize --strategy double_ma --symbols SA FG CF MA TA SR OI --n-trials 200
# -> results/optuna/DoubleMaStrategy_<ts>_best.json (best params + PBO/DSR/IS-OOS/plateau diagnostics)

# 3. Evaluate the winning params on the holdout window
python research_runner.py holdout --best results/optuna/DoubleMaStrategy_<ts>_best.json

# 4. Fit a meta-label filter: P(win) per entry signal
python research_runner.py metalabel --strategy double_ma --params-from results/optuna/DoubleMaStrategy_<ts>_best.json
# -> results/meta/DoubleMaStrategy_<ts>.joblib

# 5. Evaluate it: gated vs. baseline vs. a random-rejection baseline of the same size
python research_runner.py meta-backtest --model results/meta/DoubleMaStrategy_<ts>.joblib
```

- **Optimize** lays out anchored walk-forward folds plus a trailing `--holdout-frac` window. The objective is Sharpe penalized for too few trades, excess drawdown, and forced liquidations, averaged across folds and penalized by their standard deviation. After the search, `optimize` reports **PBO** (Probability of Backtest Overfitting), **Deflated Sharpe Ratio**, an **IS/OOS decay** ratio, and a **parameter-plateau** check.
- **Meta-labeling** trains a classifier on the features available *at* each entry signal (rolling market indicators + strategy registered indicators) to predict whether that trade will be profitable. Labels are `net_pnl > 0` and sample weight is `|net_pnl|`. Training uses **purged, embargoed walk-forward** CV, and the gating threshold is chosen from out-of-fold predictions by **expected value of the trades it would keep**. `meta-backtest` reports gated vs. baseline metrics, out-of-fold coverage, the sum of `net_pnl` on rejected trades, and where the gated result lands in a distribution of 200 random-rejection baselines at the same rejection rate.
- **What** `meta-backtest` **is actually comparing** — an entry signal only receives an out-of-fold probability if it falls inside a fold's validation window, so most bars carry no prediction at all. Those bars are passed through untouched in *both* arms, which keeps gated-vs-baseline a measurement of the filter's decisions rather than of the harness's coverage. Deployment uses the opposite policy (`on_missing='block'`) because a missing probability means the model genuinely cannot score the bar yet. The report leads with **OOF coverage** (how many of the window's signals the filter actually judged) and warns below 50%.
- Both tools are read-only with respect to the bundled example strategies and require no changes to add a new one: declare `space`/`fixed_params`/`constraints` on the `Strategy` subclass (optional) and everything else applies automatically.
