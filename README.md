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
- **Research tools** — Optuna parameter optimization with anchored walk-forward validation, a locked holdout window, and overfitting diagnostics
- **Meta-labeling** — a second-stage classifier that vetoes a strategy's weakest entries, validated by purged walk-forward with a shuffled-label control
- **Live signals** — the next session's target positions for any strategy, gated by a meta-model or not

## Quick Start

- Requirement: Python 3.11+

```bash
cd FuturesToolkit
pip install -r requirements.txt
```

- DCE needs credentials ([apply here](http://www.dce.com.cn/dce/channel/list/7000198.html)) for historical data.

```bash
export DCE_API_KEY=...
export DCE_SECRET=...
```

- Download the data. **The first run takes about an hour**.

```bash
python -m datafeed.data_update
```

- Run a backtest. Outputs (charts, trade log) land in `results/`.

```bash
python runner.py --symbols SA CF RB --start 2020-01-01 --end 2026-12-31 \
    --strategy double_ma --cash 100000
```

## Documentation

| Page | Contents |
| --- | --- |
| [Data Pipeline](docs/data.md) | Exchange downloads, `data_update` flags, generated files, product registry |
| [Backtesting](docs/backtest.md) | `runner.py` flags, four-phase execution model, outputs, metrics |
| [Writing a Strategy](docs/strategy.md) | `Strategy` lifecycle, `SetupContext` / `BarContext` API, conventions |
| [Parameter Optimization](docs/research.md) | `research_runner.py` subcommands, walk-forward splits, objective, overfitting diagnostics |
| [Meta-Labeling](docs/meta-labeling.md) | `meta_runner.py` workflow, leak controls, how to read the result |
| [Live Signals](docs/live.md) | `live_runner.py` flags, output columns, caveats |
| [Project Layout](docs/architecture.md) | Directory map, tests |
