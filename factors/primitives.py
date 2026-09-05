"""Factor operator library: time-series ops on one series, cross-sectional
ops on a panel.

The two families are shape-typed and refuse each other's input rather than
silently reshaping it: ``ts_*`` operators take a single symbol's ``float64[n_bars]``
series (the same shape ``strategies.base.SetupContext`` accessors return),
``cs_*`` operators take a ``float64[n_bars, n_symbols]`` panel (the same shape
:class:`factors.base.FactorPanel.values` is). Mixing the two up produces a
result that still *runs* -- a 1D array broadcasts against a rolling window
just fine -- so the shape check is the only thing standing between a wrong
axis and a plausible-looking factor.

``ts_return`` is lifted verbatim from
``strategies.cross_sectional_momentum._momentum`` so the bundled momentum
factor and the hand-written example strategy score identically.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

__all__ = [
    'lag', 'ts_return', 'ts_diff', 'ts_mean', 'ts_std', 'ts_zscore', 'ts_rank',
    'cs_rank', 'cs_demean', 'cs_zscore', 'winsorize',
]


def _require_series(arr, fname: str) -> np.ndarray:
    arr = np.asarray(arr, dtype='float64')
    if arr.ndim != 1:
        raise ValueError(
            f"{fname} operates on a single time series (1D); got shape {arr.shape} "
            f"-- pass one symbol's array, not a whole panel."
        )
    return arr


def _require_panel(arr, fname: str) -> np.ndarray:
    arr = np.asarray(arr, dtype='float64')
    if arr.ndim != 2:
        raise ValueError(
            f"{fname} operates on a cross-sectional panel (2D, bars x symbols); "
            f"got shape {arr.shape} -- pass a whole panel, not one symbol's series."
        )
    return arr


def _require_positive_window(window: int, fname: str) -> int:
    window = int(window)
    if window <= 0:
        raise ValueError(f"{fname}: window must be positive, got {window}")
    return window


def lag(arr, n: int) -> np.ndarray:
    """Shift ``arr`` ``n`` steps into the past along axis 0, with a NaN head.

    Works on either a series or a panel -- shifting is shape-agnostic. ``n``
    must be non-negative: a factor is not allowed to look forward, so a
    negative lag raises rather than silently reading future bars. ``n`` at or
    past the length of ``arr`` is a legitimate degenerate case (a factor's
    own warmup can exceed its data) and returns an all-NaN array rather than
    raising.
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"lag: n must not be negative (no looking forward), got {n}")
    arr = np.asarray(arr, dtype='float64')
    out = np.full_like(arr, np.nan)
    if n == 0:
        out[...] = arr
        return out
    length = arr.shape[0]
    if n >= length:
        return out
    out[n:] = arr[:-n]
    return out


def ts_return(close, lookback: int, skip: int = 0) -> np.ndarray:
    """``close[t-skip] / close[t-skip-lookback] - 1``, with a leading-NaN warmup.

    Identical formula to ``strategies.cross_sectional_momentum._momentum`` --
    lifted rather than reimplemented so the bundled momentum factor and that
    hand-written example strategy always agree.
    """
    close = _require_series(close, 'ts_return')
    out = np.full_like(close, np.nan, dtype='float64')
    span = lookback + skip
    if lookback <= 0 or len(close) <= span:
        return out
    past = close[:-span] if span else close
    recent = close[lookback:len(close) - skip] if skip else close[lookback:]
    with np.errstate(divide='ignore', invalid='ignore'):
        out[span:] = np.where(past > 0, recent / past - 1.0, np.nan)
    return out


def ts_diff(arr, n: int) -> np.ndarray:
    """``arr[t] - arr[t-n]``, with a leading-NaN warmup of length ``n``."""
    arr = _require_series(arr, 'ts_diff')
    n = _require_positive_window(n, 'ts_diff')
    shifted = lag(arr, n)
    return arr - shifted


def _rolling(arr: np.ndarray, window: int, fn) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype='float64')
    if window > n:
        return out
    for t in range(window - 1, n):
        out[t] = fn(arr[t - window + 1: t + 1])
    return out


def ts_mean(arr, window: int) -> np.ndarray:
    """Trailing, inclusive rolling mean over ``window`` bars.

    ``window`` longer than the series is a legitimate degenerate case (not
    every factor parameter fits every symbol's history) and returns an
    all-NaN array; a non-positive ``window`` is a programmer error and raises.
    """
    arr = _require_series(arr, 'ts_mean')
    window = _require_positive_window(window, 'ts_mean')
    return _rolling(arr, window, lambda w: np.mean(w) if not np.isnan(w).any() else np.nan)


def ts_std(arr, window: int) -> np.ndarray:
    """Trailing, inclusive rolling population std (ddof=0) over ``window`` bars."""
    arr = _require_series(arr, 'ts_std')
    window = _require_positive_window(window, 'ts_std')
    return _rolling(arr, window, lambda w: np.std(w) if not np.isnan(w).any() else np.nan)


def ts_zscore(arr, window: int) -> np.ndarray:
    """``(arr - ts_mean) / ts_std`` over a trailing ``window``.

    NaN on a flat window (std == 0) rather than +/-inf: a window with no
    dispersion carries no ranking information, and a divide-by-zero inf would
    otherwise dominate every downstream statistic it touches.
    """
    arr = _require_series(arr, 'ts_zscore')
    mean = ts_mean(arr, window)
    std = ts_std(arr, window)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(std > 0, (arr - mean) / std, np.nan)
    return out


def ts_rank(arr, window: int) -> np.ndarray:
    """Percentile position (``[0, 1]``, ties averaged) of ``arr[t]`` within
    its own trailing ``window``-bar history, 1.0 being the top.

    A window of one has nothing to compare against but itself, so it always
    reports 1.0 (the top) rather than an undefined mid-point.
    """
    arr = _require_series(arr, 'ts_rank')
    window = _require_positive_window(window, 'ts_rank')
    n = len(arr)
    out = np.full(n, np.nan, dtype='float64')
    if window > n:
        return out
    for t in range(window - 1, n):
        w = arr[t - window + 1: t + 1]
        if np.isnan(w[-1]):
            continue
        valid_mask = ~np.isnan(w)
        valid = w[valid_mask]
        if valid.size <= 1:
            out[t] = 1.0
            continue
        ranks = stats.rankdata(valid, method='average')
        pos = int(valid_mask[:-1].sum())
        out[t] = (ranks[pos] - 1.0) / (valid.size - 1.0)
    return out


def cs_rank(panel) -> np.ndarray:
    """Per-row (per-bar) percentile rank in ``[0, 1]``, ties sharing the
    average rank, NaN entries excluded from that row's ranking.

    A row with a single valid product scores it at the midpoint (0.5): a
    percentile over one observation carries no information about where it
    sits relative to peers.
    """
    panel = _require_panel(panel, 'cs_rank')
    n_bars, n_symbols = panel.shape
    out = np.full((n_bars, n_symbols), np.nan, dtype='float64')
    valid = ~np.isnan(panel)
    counts = valid.sum(axis=1)
    for i in np.flatnonzero(counts > 0):
        cols = np.flatnonzero(valid[i])
        if len(cols) < 2:
            out[i, cols] = 0.5
            continue
        vals = panel[i, cols]
        ranks = stats.rankdata(vals, method='average')
        out[i, cols] = (ranks - 1.0) / (len(cols) - 1.0)
    return out


def cs_demean(panel) -> np.ndarray:
    """Subtract each row's cross-sectional mean, ignoring NaN (missing)
    products rather than treating them as zero."""
    panel = _require_panel(panel, 'cs_demean')
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        row_mean = np.nanmean(panel, axis=1, keepdims=True)
    return panel - row_mean


def cs_zscore(panel) -> np.ndarray:
    """``(value - row_mean) / row_std`` per row, ignoring missing products.

    NaN for a row with no cross-sectional dispersion (every valid product
    tied), for the same reason :func:`ts_zscore` avoids dividing by zero.
    """
    panel = _require_panel(panel, 'cs_zscore')
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        row_mean = np.nanmean(panel, axis=1, keepdims=True)
        row_std = np.nanstd(panel, axis=1, keepdims=True)
        out = np.where(row_std > 0, (panel - row_mean) / row_std, np.nan)
    return out


def winsorize(panel, limit: float) -> np.ndarray:
    """Clip each row to its own ``[limit, 1 - limit]`` quantiles, ignoring NaN.

    ``limit == 0`` is a passthrough (no clipping). ``limit`` must be strictly
    less than 0.5 -- at 0.5 both quantiles coincide at the median and every
    value in the row would collapse to it.
    """
    panel = _require_panel(panel, 'winsorize')
    limit = float(limit)
    if not 0.0 <= limit < 0.5:
        raise ValueError(f"winsorize: limit must be in [0, 0.5), got {limit}")
    if limit == 0.0:
        return panel.copy()
    with np.errstate(invalid='ignore'):
        lo = np.nanquantile(panel, limit, axis=1, keepdims=True)
        hi = np.nanquantile(panel, 1.0 - limit, axis=1, keepdims=True)
    return np.clip(panel, lo, hi)
