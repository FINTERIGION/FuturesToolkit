"""Synthetic MarketData builders for fast, deterministic engine/broker tests.

Deliberately not backed by real CSV data -- these fixtures give each test
full control over sessions, contract lifecycles, and rolls without depending
on what happens to be in data/*.csv.
"""

import numpy as np

from core.market import ContractSeries, MarketData, ProductPanel

_WEIGHTED_FIELDS = ('open', 'high', 'low', 'close', 'settle', 'oi', 'volume', 'session')


def build_panel(symbol, n_bars, weighted=None, contracts=None, contract_by_bar=None, first_bar=0):
    """
    weighted: dict of field -> list[float] (missing fields default to 0.0;
        'session' defaults to all-live if omitted).
    contracts: dict code -> {bar_index: (open, high, low, close, settle, oi, volume)}
    contract_by_bar: list[str] length n_bars, '' where no contract is live.
    """
    w = weighted or {}
    arrays = {}
    for field in _WEIGHTED_FIELDS:
        if field in w:
            arrays[field] = np.asarray(w[field], dtype='float64')
        elif field == 'session':
            arrays[field] = np.ones(n_bars, dtype='float64')
        else:
            arrays[field] = np.zeros(n_bars, dtype='float64')

    series = {}
    for code, rows in (contracts or {}).items():
        idx = sorted(rows)
        live_bars = np.array(idx, dtype='int64')
        ohlcv = np.array([rows[i] for i in idx], dtype='float64')
        series[code] = ContractSeries(code=code, start=int(live_bars[0]), live_bars=live_bars, ohlcv=ohlcv)

    cbb = contract_by_bar if contract_by_bar is not None else [''] * n_bars
    return ProductPanel(
        symbol=symbol,
        weighted=arrays,
        contracts=series,
        contract_by_bar=np.array(cbb, dtype=object),
        first_bar=first_bar,
    )


def build_market(products: dict, n_bars: int):
    dates = np.array(
        [np.datetime64('2024-01-01') + np.timedelta64(i, 'D') for i in range(n_bars)],
        dtype='datetime64[D]',
    )
    return MarketData(dates=dates, products=products)


def build_trending_market(symbols=('SA',), n_bars=900, seed=0):
    """A multi-product synthetic market with a trend plus a ~15-bar cycle, so
    both crossover and mean-reversion strategies generate real trades.

    Exists so tests that exercise the layers *above* the data feed -- the web
    API's job/store/serialization plumbing, for one -- can run a real
    backtest without any of `data/*.csv`, which is gitignored and therefore
    absent on a fresh clone and in CI. Loading those CSVs has its own tests
    (`test_data_update`, `test_sources`); nothing above needs them to be real.
    """
    panels = {}
    for offset, symbol in enumerate(symbols):
        rng = np.random.default_rng(seed + offset)
        t = np.arange(n_bars, dtype='float64')
        # Offset the phase per symbol so a multi-product run is not just the
        # same series twice, which would make cross-sectional logic degenerate.
        close = 100.0 + 0.03 * t + 4.0 * np.sin(t / 15.0 + offset) + rng.normal(0, 0.3, n_bars)
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, n_bars)
        low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, n_bars)
        volume = rng.uniform(1000, 2000, n_bars)
        oi = rng.uniform(5000, 6000, n_bars)

        weighted = {
            'open': open_, 'high': high, 'low': low, 'close': close,
            'settle': close.copy(), 'oi': oi, 'volume': volume,
            'session': np.ones(n_bars),
        }
        code = f'{symbol}C1'
        contracts = {
            code: {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                   for i in range(n_bars)},
        }
        panels[symbol] = build_panel(
            symbol, n_bars, weighted=weighted, contracts=contracts,
            contract_by_bar=[code] * n_bars, first_bar=0,
        )
    return build_market(panels, n_bars)


def build_oscillating_market(symbols=('SA', 'CF'), n_bars=300):
    """Two products on a large-amplitude sine plus drift, so a crossover
    strategy changes side often enough to produce entries, exits and
    reversals within a few hundred bars.

    Distinct from `build_trending_market` in ways tests depend on: the swing
    is wide (±20 vs ±4), each bar's open equals its close so there is no
    overnight gap for a stop to trigger on, and the high/low are exactly ±1
    rather than random. That makes fills predictable, which is what the
    signal and meta-labeling suites assert against.

    The generator is seeded once for the whole call, not per symbol, so each
    product draws from the same stream in turn -- reordering `symbols` gives
    different data, and the second product's series depends on the first.
    Preserved deliberately: this reproduces the numbers the signal and
    meta-labeling suites were written against.
    """
    rng = np.random.default_rng(7)
    products = {}
    for k, sym in enumerate(symbols):
        t = np.arange(n_bars, dtype='float64')
        close = 100.0 + 20.0 * np.sin(t / (18.0 + 5 * k)) + 0.02 * t + rng.normal(0, 0.6, n_bars)
        close = np.maximum(close, 5.0)
        code = f'{sym}509'
        contracts = {code: {i: (close[i], close[i] + 1, close[i] - 1, close[i], close[i],
                                1000.0 + i, 500.0 + i) for i in range(n_bars)}}
        products[sym] = build_panel(
            sym, n_bars,
            weighted={
                'open': close, 'high': close + 1.0, 'low': close - 1.0,
                'close': close, 'settle': close,
                'oi': 1000.0 + np.arange(n_bars), 'volume': 500.0 + np.arange(n_bars),
                'session': np.ones(n_bars),
            },
            contracts=contracts,
            contract_by_bar=[code] * n_bars,
        )
    return build_market(products, n_bars)
