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
