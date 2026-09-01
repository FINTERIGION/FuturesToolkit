# FuturesToolkit

A self-built daily-bar backtesting engine for Chinese commodity futures, covering **CZCE**, **SHFE**, and **DCE**.

It downloads historical data straight from each exchange, builds open-interest–weighted daily bars for signals, executes on each product's registered main contracts, and exports equity curves, trade logs, and signal charts.

## Features

- **Data pipeline** — fetch history from three exchanges behind one interface, clean contract-level OHLC, and aggregate to OI-weighted continuous series
- **Calendar execution** — signals on the weighted series; fills on the real main contracts declared per product, and rolled automatically
- **Multi-product** — strategies loop over `ctx.symbols` and trade each independently, or trade the universe as a single cross-section
- **Futures cost model** — per-product multiplier, margin ratio, and commission from `datafeed/products.py`, with long/short-symmetric equity accounting and daily forced liquidation on a margin breach
- **Strategy API** — `Strategy` / `SetupContext` / `BarContext`, target-position order semantics (`ctx.set_target`), per-product indicator warmup skipping, protective stops
- **Metrics & reports** — Sharpe, Sortino, Calmar, max drawdown & recovery, win rate, turnover, capital exposure, per-symbol breakdown, forced-liquidation count
- **Charts** — equity, returns, position, price & signals, summary plots
- **Research tools** — Optuna parameter optimization with anchored walk-forward validation, a locked holdout window, and overfitting diagnostics



## Requirements

- Python 3.11+

```
pandas==3.0.1
numpy==2.4.2
matplotlib==3.10.8
TA-Lib==0.7.1
pytest==9.1.1
optuna==4.9.0
scipy==1.17.1
```



## Installation

```bash
cd FuturesToolkit
pip install -r requirements.txt
```



## Quick Start



### 1. Download / refresh data

Set `DCE_API_KEY` and `DCE_SECRET` before syncing DCE products ([get one here](http://www.dce.com.cn/dce/channel/list/7000198.html)):

```bash
export DCE_API_KEY=...
export DCE_SECRET=...
```

Data is pulled from the exchanges and written to `data/{SYMBOL}.csv` (contract-level bars) and `data/{SYMBOL}_weighted.csv` (OI-weighted series) for each registered product.

```bash
python -m datafeed.data_update              # incremental refresh (all products)
python -m datafeed.data_update SA CF        # selected products only
python -m datafeed.data_update --force      # re-download everything
python -m datafeed.data_update --rebuild-only
```

**First run takes a while.** SHFE and DCE ship one payload per trading day, so it takes roughly an hour.

### 2. Run a backtest

Run `python runner.py --help` for the full flag list. Outputs land in `results/` (charts, trade log).

```bash
python runner.py --symbols SA FG CF --start 2020-01-01 --end 2026-12-31 --strategy double_ma --cash 100000
```

Strategy parameters can be overridden inline or by an `optimize` report (see [Parameter Optimization](#parameter-optimization)):

```bash
python runner.py --strategy double_ma --param fast_period=7 --param slow_period=40
python runner.py --strategy double_ma --params-from results/optuna/DoubleMaStrategy_<ts>_best.json
```



### 3. Switch strategies

Built-in examples, selected via `--strategy`:

```bash
python runner.py --strategy double_ma
python runner.py --strategy rsi_mean_reversion
python runner.py --strategy cross_sectional_momentum # cross-sectional
python runner.py --strategy my_strategy              # template
```



## Project Layout

```
FuturesToolkit/
├── runner.py                  # CLI entry point
├── research_runner.py         # Research CLI: show-space / optimize / holdout
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
│   └── my_strategy.py
├── research/                  # Optuna parameter optimization (strategy-agnostic)
│   ├── space.py               #   search-space resolution (declared / inferred / CLI override)
│   ├── splits.py              #   anchored walk-forward folds + locked holdout window
│   ├── warmup.py              #   exact warmup probing (pad covers the slowest product)
│   ├── runner_api.py          #   single-window backtest with a leak-safe warmup pad
│   ├── objective.py           #   Optuna trial scoring
│   ├── optimize.py            #   study driver + holdout evaluation
│   └── overfit.py             #   PBO (CSCV), Deflated Sharpe, IS/OOS decay, plateau check
├── datafeed/                  # Data pipeline
│   ├── sources.py             #   per-exchange download & cache adapters (CZCE / SHFE / DCE)
│   ├── data_update.py         #   OI-weighted aggregation & CSV output
│   ├── data_manager.py        #   load / align / bundle data for the engine
│   ├── products.py            #   product registry (multiplier, margin, commission, roll months)
│   └── roll_calendar.py       #   date → main-month contract map (from products.py)
├── tests/                     # pytest suite (synthetic MarketData fixtures)
├── data/                      # Generated CSVs (gitignored)
├── cache/                     # Raw exchange payloads per venue: cache/{CZCE,SHFE,DCE}/ (gitignored)
└── results/                   # Backtest outputs (gitignored), incl. results/optuna
```



## Writing a Strategy

1. Subclass `Strategy` from `strategies.base`.
2. Precompute full-series indicators in `setup(ctx)` with `ctx.add_indicator`.
3. Implement trading logic in `on_bar(ctx)`.
4. Add research strategies under `strategies/`, or start from `strategies/my_strategy.py`.
5. Optionally declare a `space` dict alongside `params` to control how `research_runner.py optimize` searches this strategy's parameters.

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

Available on `BarContext`:


| Accessor                                                                          | Meaning                                                                   |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `ctx.bar(sym)`                                                                    | `Bar(open, high, low, close, settle, volume, oi)` for the weighted series |
| `ctx.ind(name, sym)`                                                              | registered indicator value at this bar                                    |
| `ctx.position(sym)`                                                               | net lots (signed)                                                         |
| `ctx.equity` / `cash` / `margin_used` / `available`                               | account state                                                             |
| `ctx.can_trade(sym)`                                                              | own warmup done + listed + real print today + mapped contract live today  |
| `ctx.contract(sym)`                                                               | today's calendar contract code, e.g. `'SA509'`                            |
| `ctx.set_target(sym, lots)` / `buy(sym, lots)` / `sell(sym, lots)` / `close(sym)` | orders                                                                    |
| `ctx.set_stop(sym, price=None, distance=None)` / `cancel_stop(sym)`               | protective stop                                                           |
| `ctx.size_for_risk(sym, stop_distance, risk_pct)`                                 | lots sized to risk `risk_pct` of equity on a stop-out                     |


Orders always fill against that day's calendar contract, so the strategy code never has to think about which physical contract an order lands on, or handle rolls itself.

## Configuration Reference


| Flag                | Meaning                                                                         | Default                     |
| ------------------- | ------------------------------------------------------------------------------- | --------------------------- |
| `--symbols`         | Products whose weighted + contract data are loaded                              | `SA FG CF BU RB HC C JM V`  |
| `--start` / `--end` | Backtest window                                                                 | `2020-01-01` / `2026-12-31` |
| `--cash`            | Starting equity (CNY)                                                           | `100000`                    |
| `--strategy`        | `double_ma` / `rsi_mean_reversion` / `cross_sectional_momentum` / `my_strategy` | `double_ma`                 |
| `--slippage`        | Fill slippage in price points                                                   | `0.0`                       |
| `--lots`            | Lots per trade                                                                  | `1`                         |
| `--update-data`     | Incremental refresh from the exchanges before running                           | off                         |
| `--keep-last N`     | Delete result files from all but the N most recent runs                         | off                         |


Multiplier, margin, commission, and main months are set in `datafeed/products.py` per product. Commission is either `commission_rate` or `commission_per_lot`. `main_months` lists the delivery months a product actually trades and `roll_lead_months` says how far ahead of delivery to leave them.

## Parameter Optimization

`research_runner.py` works on any strategy `strategies.discover_strategies()` finds. It has three subcommands:

```bash
# 1. See what will be tuned
python research_runner.py show-space --strategy double_ma

# 2. Optuna anchored walk-forward search
python research_runner.py optimize --strategy double_ma --symbols SA FG CF BU RB HC --n-trials 100
# -> results/optuna/DoubleMaStrategy_<ts>_best.json (best params + PBO/DSR/IS-OOS/plateau diagnostics)

# 3. Evaluate the winning params on the holdout window
python research_runner.py holdout --best results/optuna/DoubleMaStrategy_<ts>_best.json
```

- **Optimize** lays out anchored walk-forward folds plus a trailing `--holdout-frac` window. The objective is Sharpe penalized for too few trades, excess drawdown, and forced liquidations, averaged across folds and penalized by their standard deviation. After the search, `optimize` reports **PBO** (Probability of Backtest Overfitting), **Deflated Sharpe Ratio**, an **IS/OOS decay** ratio, and a **parameter-plateau** check.
- **Holdout** is evaluated once, on a window the search never touched, and the report records that it has been used.
