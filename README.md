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
- **Parameter tuning** — Optuna parameter optimization with anchored walk-forward validation, a locked holdout window, and overfitting diagnostics (PBO, Deflated Sharpe, IS/OOS decay, plateau check)
- **Web panel** — a local browser UI for managing products, downloading data, running backtests and Optuna tuning with live progress, and comparing run history

## Quick Start

Requirement: Python 3.11+

```bash
cd FuturesToolkit
pip install -r requirements.txt
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
python ft.py backtest --symbols SA CF RB --start 2020-01-01 --end 2026-12-31 --strategy double_ma --cash 200000
```

Tune it, then replay the winning parameters with full charts.

```bash
python ft.py optimize --strategy double_ma --symbols SA CF RB --n-trials 200
python ft.py backtest --strategy double_ma --params-from results/optuna/..._best.json
```

Or drive all of it from the browser.

```bash
python ft.py web
```

| Subcommand | What it does |
| --- | --- |
| `data` | Download exchange history, rebuild OI-weighted daily bars |
| `backtest` | Run one strategy over a date range |
| `show-space` | Print a strategy's tunable search space |
| `optimize` | Optuna anchored walk-forward parameter search |
| `holdout` | Evaluate an optimize report on its locked holdout window (once) |
| `web` | Serve the browser panel |

## Documentation

| Page | Contents |
| --- | --- |
| [Data Pipeline](docs/data.md) | Exchange downloads, `ft.py data` flags, generated files, product registry |
| [Backtesting](docs/backtest.md) | `ft.py backtest` flags, four-phase execution model, outputs, metrics |
| [Writing a Strategy](docs/strategy.md) | `Strategy` lifecycle, `SetupContext` / `BarContext` API, conventions |
| [Parameter Optimization](docs/research.md) | `ft.py optimize` / `holdout`, walk-forward splits, objective, overfitting diagnostics |
| [Web Panel](docs/web.md) | Browser UI for products, data, backtest, optimize, run history |
| [Project Layout](docs/architecture.md) | Directory map, tests |
