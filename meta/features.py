"""Feature definitions for the meta-labeling gate.

**One definition, two call sites.** ``build_feature_arrays`` computes every
feature as a full-length array from a ``SetupContext``; ``feature_row`` pulls
one row out of that same dict at a given bar. The backtest gate and the live
gate both go through ``feature_row``, and the offline sample builder in
``meta.dataset`` goes through it too -- so a feature can never mean one thing
in training and another at inference. That is the only mechanism preventing
train/serve skew here, which is why there is no second, "vectorized" path.

Features are computed on each product's OI-weighted continuous series, at the
*decision* bar (the close a signal is generated from), never on the execution
contract and never on the fill bar.

Two groups:

* ``NEUTRAL`` features describe market state without a direction -- how
  volatile, how trending, how much volume and open interest. They are read as
  they are.
* ``DIRECTIONAL`` features carry a sign that only means something relative to
  the trade being considered. They are multiplied by ``side`` (+1 long, -1
  short) at read time, which turns "the market rose 3%" into "the market moved
  3% *in the direction of this trade*". Without that flip the model would have
  to learn the direction interaction from a sample of ~1.5k trades, and it
  would spend most of its capacity doing so.

Every array is leading-NaN-only: ``core.indicators.guard`` (which
``SetupContext.add_indicator`` runs) rejects embedded NaN, and TA-Lib silently
returns all-NaN if it ever sees one. Division guards below exist for that
reason, not for cosmetics -- a dark bar is flattened to ``open == high == low
== close`` by ``datafeed.data_manager._align_contract_ohlc``, so zero ranges
and zero volumes are normal inputs, not corrupt ones.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import talib

__all__ = [
    'FEATURE_NAMES', 'NEUTRAL', 'DIRECTIONAL', 'FEATURE_PREFIX',
    'build_feature_arrays', 'feature_row',
]

FEATURE_PREFIX = '_meta_'

NEUTRAL: List[str] = [
    'atr_pct',        # ATR(14) / close -- volatility in units of price
    'vol_ratio',      # ATR(14) / ATR(60) -- is volatility expanding or contracting
    'rv20',           # 20-bar stdev of log returns
    'adx',            # ADX(14) -- trend strength, direction-free by construction
    'ibs',            # (close - low) / (high - low) -- where in the day's range we closed
    'vol_ma_ratio',   # volume / SMA(volume, 20)
    'oi_chg_5',       # oi / oi[-5] - 1
    'oi_ma_ratio',    # oi / SMA(oi, 20)
]

DIRECTIONAL: List[str] = [
    'ret_5',          # log return over 5 bars
    'ret_20',
    'ret_60',
    'dist_sma20',     # close / SMA(20) - 1
    'dist_sma60',
    'chan_pos_60',    # position in the 60-bar range, centred on 0
    'cs_mom_rank',    # cross-sectional percentile of ret_20, centred on 0
]

# Order is part of the model artifact: `meta.model.load` refuses a model whose
# stored names do not match this list, because a silently reordered feature
# vector produces plausible-looking nonsense rather than an error.
FEATURE_NAMES: List[str] = NEUTRAL + DIRECTIONAL

_ATR_FAST = 14
_ATR_SLOW = 60
_RV = 20
_ADX = 14
_SMA_FAST = 20
_SMA_SLOW = 60
_CHANNEL = 60
_VOL_MA = 20
_OI_LOOKBACK = 5
_OI_MA = 20


def _safe_div(num: np.ndarray, den: np.ndarray, fill: float) -> np.ndarray:
    """``num / den`` with non-positive/non-finite denominators replaced by ``fill``.

    ``fill`` is the feature's neutral value, not zero: a dark bar should read
    as "nothing unusual", and zero is a strong signal for a ratio centred on 1.
    """
    out = np.full(len(num), np.nan, dtype='float64')
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[ok] = num[ok] / den[ok]
    degenerate = np.isfinite(num) & np.isfinite(den) & ~ok
    out[degenerate] = fill
    return out


def _log_returns(close: np.ndarray) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype='float64')
    prev, cur = close[:-1], close[1:]
    ok = (prev > 0) & (cur > 0)
    out[1:][ok] = np.log(cur[ok] / prev[ok])
    out[1:][~ok] = 0.0
    return out


def _ret_n(close: np.ndarray, n: int) -> np.ndarray:
    """Log return over ``n`` bars, leading NaN for the first ``n``."""
    out = np.full(len(close), np.nan, dtype='float64')
    if len(close) <= n:
        return out
    prev, cur = close[:-n], close[n:]
    ok = (prev > 0) & (cur > 0)
    tail = out[n:]
    tail[ok] = np.log(cur[ok] / prev[ok])
    tail[~ok] = 0.0
    return out


def _oi_change(oi: np.ndarray, lookback: int) -> np.ndarray:
    """``oi[t] / oi[t - lookback] - 1``, with a leading-NaN warmup.

    Same shape as ``strategies.finter_momentum._oi_change``; duplicated rather
    than imported because that module is a private, gitignored strategy and
    this package must import cleanly without it.
    """
    out = np.full(len(oi), np.nan, dtype='float64')
    if lookback <= 0 or len(oi) <= lookback:
        return out
    prev, cur = oi[:-lookback], oi[lookback:]
    out[lookback:] = np.where(prev > 0, cur / np.where(prev > 0, prev, 1.0) - 1.0, 0.0)
    return out


def _first_valid(arr: np.ndarray) -> int:
    mask = np.isfinite(arr)
    return int(np.argmax(mask)) if mask.any() else len(arr)


def _sanitize(arr: np.ndarray, fill: float) -> np.ndarray:
    """Leave the leading NaN run alone; make everything after it finite.

    ``+/-inf`` and any NaN that survived the division guards become ``fill``.
    Anything reaching here is a degenerate bar (flat range, zero volume), not
    a bug -- but ``guard`` would reject it and TA-Lib downstream would go
    all-NaN, so it has to be resolved one way or the other.
    """
    out = np.asarray(arr, dtype='float64').copy()
    start = _first_valid(out)
    if start >= len(out):
        return out
    tail = out[start:]
    tail[~np.isfinite(tail)] = fill
    return out


def build_feature_arrays(ctx) -> Dict[Tuple[str, str], np.ndarray]:
    """``{(feature_name, symbol): float64[n_bars]}`` for every registered feature.

    ``ctx`` is a ``strategies.base.SetupContext``. Cross-sectional features
    need the whole panel, so this computes every symbol in one pass rather
    than offering a per-symbol entry point.
    """
    symbols = list(ctx.symbols)
    arrays: Dict[Tuple[str, str], np.ndarray] = {}
    ret20_panel: Dict[str, np.ndarray] = {}

    for sym in symbols:
        high, low, close = ctx.high(sym), ctx.low(sym), ctx.close(sym)
        volume, oi = ctx.volume(sym), ctx.oi(sym)

        atr_fast = talib.ATR(high, low, close, _ATR_FAST)
        atr_slow = talib.ATR(high, low, close, _ATR_SLOW)
        sma_fast = talib.SMA(close, _SMA_FAST)
        sma_slow = talib.SMA(close, _SMA_SLOW)
        chan_hi = talib.MAX(close, _CHANNEL)
        chan_lo = talib.MIN(close, _CHANNEL)
        ret20 = _ret_n(close, 20)
        ret20_panel[sym] = ret20

        feats = {
            'atr_pct': _safe_div(atr_fast, close, 0.0),
            'vol_ratio': _safe_div(atr_fast, atr_slow, 1.0),
            'rv20': talib.STDDEV(_log_returns(close), _RV),
            'adx': talib.ADX(high, low, close, _ADX),
            'ibs': _safe_div(close - low, high - low, 0.5),
            'vol_ma_ratio': _safe_div(volume, talib.SMA(volume, _VOL_MA), 1.0),
            'oi_chg_5': _oi_change(oi, _OI_LOOKBACK),
            'oi_ma_ratio': _safe_div(oi, talib.SMA(oi, _OI_MA), 1.0),
            'ret_5': _ret_n(close, 5),
            'ret_20': ret20,
            'ret_60': _ret_n(close, 60),
            'dist_sma20': _safe_div(close, sma_fast, 1.0) - 1.0,
            'dist_sma60': _safe_div(close, sma_slow, 1.0) - 1.0,
            # Centred on 0 so that multiplying by `side` is a reflection about
            # mid-range rather than about the bottom of the channel.
            'chan_pos_60': _safe_div(close - chan_lo, chan_hi - chan_lo, 0.5) - 0.5,
        }
        neutral_fill = {'vol_ratio': 1.0, 'ibs': 0.5, 'vol_ma_ratio': 1.0, 'oi_ma_ratio': 1.0}
        for name, arr in feats.items():
            arrays[(name, sym)] = _sanitize(arr, neutral_fill.get(name, 0.0))

    for sym, arr in _cross_sectional_rank(symbols, ret20_panel).items():
        arrays[('cs_mom_rank', sym)] = arr

    return arrays


def _cross_sectional_rank(symbols: List[str], panel: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-bar percentile of each symbol's 20-bar return across the universe,
    centred on 0 so the sign survives the ``side`` flip.

    Symbols whose own ``ret_20`` is not yet valid are excluded from that bar's
    ranking rather than ranked as zero -- a product still in warmup should not
    push the rest of the universe up the scale. Their own value stays NaN,
    which keeps their leading-NaN run intact. Bars with fewer than two ranked
    symbols score everyone at the midpoint: a percentile over one observation
    carries no information.
    """
    n_bars = len(next(iter(panel.values()))) if panel else 0
    stacked = np.vstack([panel[s] for s in symbols]) if symbols else np.zeros((0, n_bars))
    valid = np.isfinite(stacked)
    out = np.full(stacked.shape, np.nan, dtype='float64')

    counts = valid.sum(axis=0)
    for i in np.flatnonzero(counts > 0):
        rows = np.flatnonzero(valid[:, i])
        if len(rows) < 2:
            out[rows, i] = 0.0
            continue
        vals = stacked[rows, i]
        order = np.argsort(np.argsort(vals))
        out[rows, i] = order / (len(rows) - 1.0) - 0.5

    return {sym: out[k] for k, sym in enumerate(symbols)}


def feature_row(
    arrays: Dict[Tuple[str, str], np.ndarray], sym: str, i: int, side: int
) -> np.ndarray:
    """One feature vector in ``FEATURE_NAMES`` order, or ``None`` if any value
    at bar ``i`` is missing.

    ``side`` is the direction of the trade being judged (+1 long, -1 short);
    ``DIRECTIONAL`` features are multiplied by it. Returning ``None`` rather
    than an imputed row is deliberate: the caller decides what a missing
    feature means (``meta.dataset`` drops the sample, ``meta.filter`` lets the
    order through unfiltered), and those are different answers.
    """
    row = np.empty(len(FEATURE_NAMES), dtype='float64')
    sign = 1.0 if side >= 0 else -1.0
    for k, name in enumerate(FEATURE_NAMES):
        arr = arrays.get((name, sym))
        if arr is None or i < 0 or i >= len(arr):
            return None
        value = arr[i]
        if not np.isfinite(value):
            return None
        row[k] = value * sign if name in _DIRECTIONAL_SET else value
    return row


_DIRECTIONAL_SET = frozenset(DIRECTIONAL)
