# FuturesBacktest

A backtesting framework for **Zhengzhou Commodity Exchange** futures — **SA (soda ash)**, **FG (glass)**, and **CF (cotton)** — built on [Backtrader](https://www.backtrader.com/).

It downloads historical data from the exchange, builds open-interest–weighted daily bars for **signals**, executes on calendar contracts (January / May / September), and exports equity curves, trade logs, and signal charts.

## Features

- **Data pipeline** — fetch CZCE history per product, clean contract-level OHLC, and aggregate to OI-weighted continuous series
- **Calendar execution** — signals on the weighted series; fills on real 01/05/09 contracts (Dec–Mar → May, Apr–Jul → Sep, Aug–Nov → next Jan), with automatic rolls at the open on the first session of Apr / Aug / Dec
- **Multi-product** — load SA / FG / CF together; single-product strategies pick one via `symbol`, multi-product strategies pass `symbol=` on each order
- **Futures cost model** — per-product multiplier, margin ratio, and commission from `products.py`
- **Strategy API** — inherit `FuturesStrategyBase` for buy/sell/close helpers and signal logging
- **Metrics & reports** — Sharpe, max drawdown, win rate, trade CSV, optional R-multiple alpha report
- **Charts** — equity, returns, position, price & signals, summary plots
- **Private strategies** — keep research code under `backtest/strategies/`

## Requirements

- Python 3.8+

```
backtrader
pandas
matplotlib
numpy
```

## Installation

```bash
cd FuturesBacktest
pip install -r requirements.txt
```

## Quick Start

### 1. Download / refresh data

Data is pulled from CZCE and written to `data/{SA,FG,CF}.csv` and `data/{SA,FG,CF}_weighted.csv`. Historical years are cached under `cache/` and reused; only the current year is re-downloaded by default.

```bash
python backtest/data_update.py              # incremental refresh (all products)
python backtest/data_update.py FG CF        # selected products only
python backtest/data_update.py --force      # re-download every year
python backtest/data_update.py --rebuild-only
```

Or enable refresh when running a backtest by setting `UPDATE_DATA = True` in `backtest/main.py`.

### 2. Run a backtest

Edit the configuration block at the top of `backtest/main.py` (`SYMBOLS`, strategy, dates, cash, commission, margin), then:

```bash
python backtest/main.py
```

Outputs land in `backtest/results/` (charts, trade log, and optional alpha CSV).

### 3. Switch strategies

Public examples:

```python
from strategies.double_ma import DoubleMaStrategy
from strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from strategies.my_strategy import MyStrategy

STRATEGY = DoubleMaStrategy
STRATEGY_PARAMS = {
    'symbol': 'SA',   # or 'FG' / 'CF'
}
```

To load only one product’s feeds:

```python
SYMBOLS = ['FG']
STRATEGY_PARAMS = {'symbol': 'FG'}
```

## Project Layout

```
FuturesBacktest/
├── requirements.txt
├── LICENSE
├── data/                     # Generated CSVs (gitignored)
├── cache/                    # CZCE yearly raw files (gitignored)
└── backtest/
    ├── main.py               # Backtest entry point & config
    ├── products.py           # SA / FG / CF registry (multiplier, margin, commission)
    ├── data_update.py        # CZCE download & OI-weighted aggregation
    ├── data_manager.py       # Load / filter / feed data into Backtrader
    ├── roll_calendar.py      # Month → Jan/May/Sep contract map
    ├── backtest_engine.py    # Cerebro runner, analyzers, metrics
    ├── plotting.py           # Chart generation
    ├── strategies/           # Base, examples (tracked) + private modules (gitignored)
    └── results/              # Backtest outputs (gitignored)
```

## Writing a Strategy

1. Subclass `FuturesStrategyBase` from `strategies.base`.
2. Define indicators in `__init__` and logic in `next()`.
3. Use `buy_signal()`, `sell_signal()`, `close_signal()`, and `get_position_size()`.
4. Add research strategies under `backtest/strategies/` (one class per file), or start from `strategies/my_strategy.py`.

Minimal single-product sketch:

```python
from strategies.base import FuturesStrategyBase
import backtrader.indicators as btind

class MyStrategy(FuturesStrategyBase):
    params = (('symbol', 'SA'), ('period', 20),)

    def __init__(self):
        super().__init__()
        self.sma = btind.SMA(self.data.close, period=self.p.period)

    def next(self):
        if self._pending_order:
            return
        pos = self.get_position_size()
        if pos == 0 and self.data.close[0] > self.sma[0]:
            self.buy_signal()
        elif pos > 0 and self.data.close[0] < self.sma[0]:
            self.close_signal()
```

Switch the traded product with `params.symbol` or `STRATEGY_PARAMS['symbol']` (`'SA'`, `'FG'`, or `'CF'`). `self.data` is that product’s weighted series.

Multi-product (annotate the product on each order):

```python
for sym in self.symbols:
    if self.has_pending(sym):
        continue
    pos = self.get_position_size(sym)
    close = self.get_weighted(sym).close[0]
    if pos == 0 and close > self.sma[sym][0]:
        self.buy_signal(symbol=sym)
```

Available bar fields on a weighted feed: `open`, `high`, `low`, `close`, `volume`, `openinterest` (OI), and custom line `settle`. Every real contract in the window is on the strategy as `self.products['FG'].contracts['FG2505']` or `self.get_contract('FG2505')`.

Orders from `buy_signal()` / `sell_signal()` / `close_signal()` are routed to that product’s calendar contract when `EXECUTE_ON_CONTRACTS = True`. Rolls are handled per product in the base class; do not implement them in `next()`.

## Configuration Reference

| Parameter | Meaning | Typical default |
|-----------|---------|-----------------|
| `SYMBOLS` | Products whose weighted + contract feeds are loaded | `['SA', 'FG', 'CF']` |
| `STRATEGY_PARAMS['symbol']` | Default product for single-product strategies | `'SA'` |
| `START_DATE` / `END_DATE` | Backtest window | `2020-01-01` |
| `INITIAL_CASH` | Starting equity (CNY) | `100000` |
| `TRADE_SIZE` | Lots per trade (if strategy uses it) | `1` |
| `SLIPPAGE` | Fill slippage in price points | `0.0` |
| `EXECUTE_ON_CONTRACTS` | Weighted signals, real-contract fills | `True` |
| `UPDATE_DATA` | Incremental refresh from CZCE | `False` |
| `STRATEGY_PARAMS` | Override strategy `params` | `{}` |

Multiplier, margin, and commission are **not** set in `main.py`. Edit `backtest/products.py` per product (SA/FG multiplier 20, CF 5). Commission is either `commission_rate` (fraction of notional) or `commission_per_lot` (fixed CNY per lot).

## Data Notes

- Source: [CZCE](http://www.czce.com.cn/) historical futures files for **SA**, **FG**, and **CF**.
- Contract-level history is cleaned and saved as `data/{symbol}.csv`.
- Daily continuous series uses open-interest weighting of **all listed contracts** that day → `data/{symbol}_weighted.csv` (default `self.data` for the traded product).
- The strategy is given every loaded product’s weighted series plus **all real contracts** that print in the backtest window (`datas[0]` = default product weighted, `self.products[sym]` = that product).
- Execution (when `EXECUTE_ON_CONTRACTS` is True), **per product**:
  - Dec–Mar trade the **May** contract, Apr–Jul the **September** contract, Aug–Nov the **next January** contract (glass lists all 12 months and cotton lists odd months; fills still use 01/05/09 only).
  - Roll on the first trading day of April, August, and December: close the old contract at that day's **open**, open the new contract at its **open** (commission on both legs).
  - A signal placed the session before a roll is cancelled and re-routed onto the **new** contract so it fills at that open.
  - Any leftover lots on a non-calendar feed are swept to the target contract at the next session open.
  - Each roll closes a Backtrader trade, so win rate / expectancy count the roll-out as a completed trade.
  - Set `EXECUTE_ON_CONTRACTS = False` in `backtest/main.py` to fill on the weighted series instead.
- Trade CSV includes a `symbol` column. Charts (price / signals / position) use the **default** product.
