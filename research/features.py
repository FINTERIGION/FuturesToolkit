"""Causal, strategy-agnostic feature construction for meta-labeling.

Three layers, all indexed by bar position on a product's OI-weighted series:

  A. A small registry of generic market features (returns, volatility,
     trend strength, distance-from-extremes) that apply to any product.
  B. Automatic harvesting of whatever indicators the strategy itself
     registered via ``ctx.add_indicator`` in ``setup()``, each turned into
     a rolling z-score -- this is what lets meta-labeling pick up the most
     relevant signal for *any* strategy without knowing what it computes.
  C. Signal-context features derived from the strategy's own trading
     history (time since last signal, rolling win rate of already-closed
     trades, whether a position is currently open) -- optional, since they
     require a full engine run's events/positions, not just ``setup()``.

Every feature is causal by construction: normalization is rolling or
expanding only, never a whole-sample mean/std. The raw OHLCV inputs are
run through ``core.indicators.guard`` (an embedded NaN there means a real
data-alignment bug). Derived statistical columns are *not* guard()-ed the
same way -- a rolling z-score legitimately can't have a value until its
window fills, and a rare zero-variance window can reintroduce a NaN mid-
series without indicating a bug -- that's exactly what the meta-labeling
pipeline's ``SimpleImputer`` (see ``research/metalabel.py``) exists to
absorb. Each column is only checked for gross corruption (wrong length,
non-finite explosion).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import talib

from core.engine import Engine
from core.indicators import guard
from core.market import MarketData
from strategies.base import SetupContext

_MARKET_FEATURES: dict = {}


def market_feature(name: str):
    def register(fn):
        _MARKET_FEATURES[name] = fn
        return fn
    return register


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(arr)
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std().replace(0, np.nan)
    return ((s - mean) / std).to_numpy(dtype='float64')


@market_feature('ret_5')
def _ret_5(o, h, l, c, v, oi):
    return pd.Series(c).pct_change(5).to_numpy(dtype='float64')


@market_feature('ret_20')
def _ret_20(o, h, l, c, v, oi):
    return pd.Series(c).pct_change(20).to_numpy(dtype='float64')


@market_feature('atr14_pct')
def _atr14_pct(o, h, l, c, v, oi):
    atr = talib.ATR(h, l, c, timeperiod=14)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(c > 0, atr / c, np.nan)


@market_feature('adx_14')
def _adx_14(o, h, l, c, v, oi):
    return talib.ADX(h, l, c, timeperiod=14)


@market_feature('rsi_14')
def _rsi_14(o, h, l, c, v, oi):
    return talib.RSI(c, timeperiod=14)


@market_feature('ma_dist_50')
def _ma_dist_50(o, h, l, c, v, oi):
    sma = talib.SMA(c, timeperiod=50)
    atr = talib.ATR(h, l, c, timeperiod=14)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(atr > 0, (c - sma) / atr, np.nan)


@market_feature('vol_z_60')
def _vol_z_60(o, h, l, c, v, oi):
    return _rolling_zscore(v, 60)


@market_feature('dist_from_high_60')
def _dist_from_high_60(o, h, l, c, v, oi):
    roll_max = pd.Series(h).rolling(60, min_periods=60).max().to_numpy(dtype='float64')
    atr = talib.ATR(h, l, c, timeperiod=14)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(atr > 0, (roll_max - c) / atr, np.nan)


def harvest_strategy_indicators(market: MarketData, strategy_cls: type, params: dict) -> dict:
    """Run just ``strategy.setup()`` and return every indicator it
    registered, keyed by ``(name, symbol)`` -- the same dict
    ``BarContext.ind`` reads from at trade time. This is what makes Layer B
    strategy-agnostic: whatever a strategy computes for its own signals is
    automatically available as a meta-label feature.
    """
    strategy = strategy_cls(**params)
    engine = Engine(market, strategy)
    strategy.setup(SetupContext(engine))
    return dict(engine.indicators)


def _pairwise_diff_features(harvested: dict, sym: str, atr: np.ndarray) -> dict:
    names = sorted({name for (name, s) in harvested if s == sym})
    out = {}
    if 1 < len(names) <= 4:
        for a, b in combinations(names, 2):
            with np.errstate(divide='ignore', invalid='ignore'):
                out[f'diff_{a}_{b}'] = np.where(
                    atr > 0, (harvested[(a, sym)] - harvested[(b, sym)]) / atr, np.nan,
                )
    return out


def _bars_since(n_bars: int, marked_bars: list) -> np.ndarray:
    out = np.full(n_bars, np.nan, dtype='float64')
    marked = set(marked_bars)
    last = None
    for t in range(n_bars):
        if last is not None:
            out[t] = t - last
        if t in marked:
            last = t
    return out


def rolling_trade_winrate(n_bars: int, events: list, lookback: int = 20) -> np.ndarray:
    """``out[t]`` = win rate of the most recent (up to ``lookback``) trades
    whose ``close_bar`` is strictly before bar ``t``. NaN until the first
    trade has closed. Strictly causal: a trade still open at bar ``t``
    (even one opened long before ``t``) never contributes, since its
    outcome isn't known yet at ``t``.
    """
    out = np.full(n_bars, np.nan, dtype='float64')
    closed = sorted((e for e in events if e.get('close_bar') is not None), key=lambda e: e['close_bar'])
    wins: list = []
    j = 0
    for t in range(n_bars):
        while j < len(closed) and closed[j]['close_bar'] < t:
            wins.append(1.0 if closed[j]['net_pnl'] > 0 else 0.0)
            j += 1
        if wins:
            out[t] = float(np.mean(wins[-lookback:]))
    return out


def build_feature_matrix(
    market: MarketData,
    strategy_cls: type,
    params: dict,
    symbols: list = None,
    *,
    events: list = None,
    position_flags: dict = None,
    zscore_window: int = 252,
    winrate_lookback: int = 20,
) -> dict:
    """Return ``{symbol: pd.DataFrame(index=bar, columns=feature_name)}``.

    ``events`` (optional): the full list of signal events from
    ``research.metalabel.extract_events`` over this same ``market`` --
    enables Layer C's ``bars_since_signal`` / ``recent_winrate``. Each event
    dict needs ``symbol``, ``signal_bar``, ``close_bar``, ``net_pnl``.

    ``position_flags`` (optional): ``{symbol: np.ndarray[n_bars]}`` of 0/1,
    already aligned to ``market``'s bar index -- enables ``has_position``.
    Building this alignment is the caller's job (it already tracks the
    window/bar bookkeeping needed to do it correctly).
    """
    symbols = symbols or market.symbols
    harvested = harvest_strategy_indicators(market, strategy_cls, params)
    n = market.n_bars

    frames = {}
    for sym in symbols:
        panel = market.products[sym]
        o = guard(panel.weighted['open'], name=f'{sym}.open')
        h = guard(panel.weighted['high'], name=f'{sym}.high')
        l = guard(panel.weighted['low'], name=f'{sym}.low')
        c = guard(panel.weighted['close'], name=f'{sym}.close')
        v = guard(panel.weighted['volume'], name=f'{sym}.volume')
        oi = guard(panel.weighted['oi'], name=f'{sym}.oi')

        cols = {name: fn(o, h, l, c, v, oi) for name, fn in _MARKET_FEATURES.items()}

        atr = talib.ATR(h, l, c, timeperiod=14)
        for (name, s), arr in harvested.items():
            if s == sym:
                cols[f'ind_{name}'] = _rolling_zscore(arr, zscore_window)
        cols.update(_pairwise_diff_features(harvested, sym, atr))

        if events is not None:
            sym_events = [e for e in events if e['symbol'] == sym]
            cols['bars_since_signal'] = _bars_since(n, [e['signal_bar'] for e in sym_events])
            cols['recent_winrate'] = rolling_trade_winrate(n, sym_events, lookback=winrate_lookback)

        if position_flags is not None and sym in position_flags:
            cols['has_position'] = position_flags[sym].astype('float64')

        for name, arr in cols.items():
            arr = np.asarray(arr, dtype='float64')
            if len(arr) != n:
                raise ValueError(f"Feature {name!r}/{sym} has length {len(arr)}, expected {n}")
            cols[name] = arr

        frames[sym] = pd.DataFrame(cols, index=np.arange(n))

    return frames
