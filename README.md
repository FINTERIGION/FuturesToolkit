# FuturesToolkit

A self-built daily-bar backtesting engine for Chinese commodity futures, covering **CZCE**, **SHFE**, and **DCE**.

It downloads historical data straight from each exchange, builds open-interest–weighted daily bars for signals, executes on each product's registered main contracts, and exports equity curves, trade logs, and signal charts.

## Features

- **Data pipeline** — fetch history from three exchanges behind one interface, clean contract-level OHLC, and aggregate to OI-weighted continuous series
- **Calendar execution** — signals on the weighted series; fills on the real main contracts declared per product, and rolled automatically
- **Multi-product** — strategies loop over `ctx.symbols` and trade each independently, or trade the universe as a single cross-section
- **Futures cost model** — per-product multiplier, margin ratio, and commission from `datafeed/products.py`, with long/short-symmetric equity accounting and daily forced liquidation on a margin breach
- **Strategy API** — `Strategy` / `SetupContext` / `BarContext`, target-position order semantics (`ctx.set_target`), per-product indicator warmup skipping, protective stop/take-profit brackets
- **Metrics & reports** — Sharpe, Sortino, Calmar, max drawdown & recovery, win rate, turnover, capital exposure, per-symbol and per-exit-reason breakdowns, forced-liquidation count
- **Charts** — equity, returns, position, price & signals, summary plots
- **Overfitting checks** — validate a chosen parameter set against anchored walk-forward folds, sub-period stability, parameter sensitivity, a block bootstrap, PBO (CSCV) and the Deflated Sharpe
- **Web panel** — a local browser UI for managing products, downloading data, running backtests with live progress, and browsing run history

## Quick Start

Requirement: Python 3.11+

```bash
cd FuturesToolkit
pip install -r requirements.txt
```

Skip this if you only use the CLI.

```bash
pip install -e ".[web]"
```

DCE needs credentials ([apply here](http://www.dce.com.cn/dce/channel/list/7000198.html)) for historical data.

```bash
export DCE_API_KEY=...
export DCE_SECRET=...
```

Download the data. **The first run takes about an hour**.

```bash
python ft.py data
```

Run a backtest. Outputs (charts, trade log) land in `results/`.

```bash
python ft.py backtest --symbols SA CF FG --start 2020-01-01 --end 2026-12-31 --strategy double_ma --cash 100000
```

Or drive it from the browser.

```bash
python ft.py web
```

| Subcommand   | What it does                                                    |
| ------------ | --------------------------------------------------------------- |
| `data`       | Download exchange history, rebuild OI-weighted daily bars       |
| `backtest`   | Run one strategy over a date range                              |
| `show-space` | Print the ranges a strategy's params are scanned over            |
| `validate`   | Check one parameter set for overfitting                         |
| `web`        | Serve the browser panel                                         |

## Documentation

| Page                                       | Contents                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------- |
| [Data Pipeline](docs/data.md)              | Exchange downloads, `ft.py data` flags, generated files, product registry             |
| [Backtesting](docs/backtest.md)            | `ft.py backtest` flags, four-phase execution model, outputs, metrics                  |
| [Writing a Strategy](docs/strategy.md)     | `Strategy` lifecycle, `SetupContext` / `BarContext` API, conventions                  |
| [Overfitting Checks](docs/validation.md)   | `ft.py validate`, walk-forward and sub-period splits, sensitivity, bootstrap, PBO, DSR |
| [Web Panel](docs/web.md)                   | Browser UI for products, data, backtest, run history                                  |
| [Project Layout](docs/architecture.md)     | Directory map, tests                                                                  |

