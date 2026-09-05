"""Factor evaluation: forward returns, IC, quantile buckets, turnover, decay.

Judges a factor's predictive content directly, without running a backtest --
no sizing, no margin, no commission in the way. Every function takes plain
arrays or a :class:`factors.base.FactorPanel` and returns plain dicts/arrays;
nothing here imports a concrete factor, so it applies unchanged to any factor
``factors.discover_factors()`` finds.

Two conventions are load-bearing and easy to get wrong:

**The validity mask is ``tradable``, not ``notna``.** ``data_manager``
forward-fills prices onto bars where a product had no print and flattens
O/H/L onto the filled close, marking them only with ``session == 0``. Such a
bar looks like a perfectly good quote that returned exactly 0%. Averaging
those in silently shrinks every statistic here toward zero, so both endpoints
of every return are gated on :func:`core.market.tradable_mask` -- the same
array behind ``FactorPanel.tradable``.

**Forward returns are lagged by one bar by default.** The engine reads a
factor in ``on_bar`` and fills at the *next* open, so a factor scored on bar
``t`` cannot capture bar ``t``'s own move. ``lag=1`` lines the measurement up
with what the engine could actually have traded; ``lag=0`` is the textbook
(alphalens) convention and will read higher for any factor built from
same-bar prices.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from core.market import OHLCV_COLS, tradable_mask
from factors.base import FactorPanel
from factors.primitives import cs_rank

logger = logging.getLogger(__name__)

__all__ = [
    'forward_returns', 'panel_forward_returns', 'ic_series', 'ic_summary',
    'ic_decay', 'quantile_returns', 'cumulative_curves', 'factor_turnover',
    'factor_autocorr', 'coverage_summary', 'annual_slices',
    'rowwise_corr', 'joint_ranks',
]

TRADING_DAYS_PER_YEAR = 244.0
_CLOSE = OHLCV_COLS.index('close')
_SETTLE = OHLCV_COLS.index('settle')


def _weighted_close(market, symbols: Sequence[str]) -> np.ndarray:
    """``float64[n_bars, len(symbols)]`` of the OI-weighted continuous close."""
    return np.column_stack([market.products[s].weighted['close'] for s in symbols])


def _exec_close(market, symbols: Sequence[str]):
    """Per-bar close of the *calendar contract* actually held, plus its code.

    Returns ``(close[n_bars, n_symbols], codes[n_bars, n_symbols])``. NaN /
    ``''`` where no calendar contract printed that bar.
    """
    n_bars = market.n_bars
    close = np.full((n_bars, len(symbols)), np.nan, dtype='float64')
    codes = np.full((n_bars, len(symbols)), '', dtype=object)
    for j, sym in enumerate(symbols):
        panel = market.products[sym]
        contract_by_bar = panel.contract_by_bar
        for i in range(n_bars):
            code = contract_by_bar[i]
            if not code:
                continue
            series = panel.contracts.get(code)
            if series is None:
                continue
            row = series.row_at(i)
            if row is None:
                continue
            close[i, j] = row[_CLOSE]
            codes[i, j] = code
    return close, codes


def forward_returns(
    market,
    horizons: Sequence[int],
    lag: int = 1,
    source: str = 'weighted',
    symbols: Optional[Sequence[str]] = None,
) -> Dict[int, np.ndarray]:
    """``{horizon: float64[n_bars, n_symbols]}`` of returns realizable from a
    signal scored on each bar.

    For horizon ``h``, entry ``[t, j]`` is the return from bar ``t + lag`` to
    bar ``t + lag + h``, i.e. what a signal read on bar ``t`` would have
    earned. NaN wherever the window runs off the end of the data or either
    endpoint was not tradable.

    ``source='weighted'`` (default) measures on the OI-weighted continuous
    series -- gap-free and roll-free, which is what makes it the right series
    for judging a *signal*. ``source='exec'`` measures on the calendar
    contract the engine would really have held, and returns NaN for any
    window that spans a roll: the two contracts' prices are not comparable,
    and differencing them would book the calendar spread as a return. That
    makes ``exec`` the stricter, more realizable read, at the cost of
    dropping observations near every roll.
    """
    if lag < 0:
        raise ValueError(f"forward_returns: lag must not be negative, got {lag}")
    if source not in ('weighted', 'exec'):
        raise ValueError(f"forward_returns: unknown source {source!r}; expected 'weighted' or 'exec'")

    syms = list(symbols) if symbols is not None else list(market.symbols)
    n_bars = market.n_bars
    tmask = tradable_mask(market, syms)

    if source == 'weighted':
        close = _weighted_close(market, syms)
        codes = None
    else:
        close, codes = _exec_close(market, syms)

    out: Dict[int, np.ndarray] = {}
    for h in horizons:
        h = int(h)
        result = np.full((n_bars, len(syms)), np.nan, dtype='float64')
        entry, exit_ = lag, lag + h
        length = n_bars - exit_
        if length > 0:
            entry_px = close[entry:entry + length]
            exit_px = close[exit_:exit_ + length]
            valid = tmask[entry:entry + length] & tmask[exit_:exit_ + length] & (entry_px > 0)
            if source == 'exec':
                entry_codes = codes[entry:entry + length]
                exit_codes = codes[exit_:exit_ + length]
                valid = valid & (entry_codes == exit_codes) & (entry_codes != '')
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(valid, exit_px / entry_px - 1.0, np.nan)
            result[:length] = ratio
        out[h] = result
    return out


def panel_forward_returns(panel: FactorPanel, market, horizons: Sequence[int]) -> Dict[int, np.ndarray]:
    """:func:`forward_returns` for the products ``panel`` scored, in the
    panel's own column order.

    Prefer this over calling :func:`forward_returns` with a separately
    assembled symbol list. Everything below pairs ``panel.values[t, j]`` with
    ``fwd[t, j]`` and can only verify that the two *shapes* agree -- not that
    column ``j`` is the same product in both. A panel computed over a subset,
    or over a reordered universe, would then be silently correlated against
    another product's returns. Going through here makes that unrepresentable.
    """
    return forward_returns(market, horizons, symbols=panel.symbols)


def _check_aligned(panel: FactorPanel, fwd: np.ndarray) -> None:
    """Shape check with the alignment contract spelled out in the message.

    Shape is all that can be checked -- see :func:`panel_forward_returns` for
    why matching shapes are necessary but not sufficient.
    """
    expected = (panel.n_bars, panel.n_symbols)
    if fwd.shape != expected:
        raise ValueError(
            f"panel is {expected}, fwd is {fwd.shape}. Build fwd with "
            f"panel_forward_returns(panel, market, horizons) to keep the two "
            f"aligned -- matching shapes alone do not guarantee matching symbol order."
        )


def rowwise_corr(a: np.ndarray, b: np.ndarray, min_names: int = 3) -> np.ndarray:
    """Per-row Pearson correlation over the cells valid in *both* arrays."""
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    valid = ~np.isnan(a) & ~np.isnan(b)
    counts = valid.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        n = np.maximum(counts, 1)
        a0 = np.where(valid, a, 0.0)
        b0 = np.where(valid, b, 0.0)
        mean_a = a0.sum(axis=1) / n
        mean_b = b0.sum(axis=1) / n
        da = np.where(valid, a - mean_a[:, None], 0.0)
        db = np.where(valid, b - mean_b[:, None], 0.0)
        cov = (da * db).sum(axis=1)
        var_a = (da * da).sum(axis=1)
        var_b = (db * db).sum(axis=1)
        denom = np.sqrt(var_a * var_b)
        corr = np.where(denom > 0, cov / denom, np.nan)
    return np.where(counts >= min_names, corr, np.nan)


def joint_ranks(a: np.ndarray, b: np.ndarray):
    """Rank both arrays row-wise over their *jointly* valid cells.

    Ranking each array over its own valid cells would compare two different
    cross-sections whenever the factor and the return disagree on which
    products are usable, so the masks are intersected first.
    """
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    joint_invalid = np.isnan(a) | np.isnan(b)
    a2 = np.where(joint_invalid, np.nan, a)
    b2 = np.where(joint_invalid, np.nan, b)
    ra = stats.rankdata(a2, method='average', axis=1, nan_policy='omit')
    rb = stats.rankdata(b2, method='average', axis=1, nan_policy='omit')
    return ra, rb


def ic_series(panel: FactorPanel, fwd: np.ndarray, method: str = 'spearman', min_names: int = 3) -> np.ndarray:
    """``float64[n_bars]``: the cross-sectional correlation between the factor
    and the forward return on each bar.

    ``spearman`` (default) is rank IC -- robust to a single outlier product
    dominating a bar. ``pearson`` is the raw-value IC. Bars with fewer than
    ``min_names`` products valid in both the factor and the return give NaN:
    a correlation over two points is always +/-1 and would otherwise flood
    the average with noise.
    """
    _check_aligned(panel, fwd)
    if method == 'spearman':
        ra, rb = joint_ranks(panel.values, fwd)
        return rowwise_corr(ra, rb, min_names)
    if method == 'pearson':
        return rowwise_corr(panel.values, fwd, min_names)
    raise ValueError(f"ic_series: unknown method {method!r}; expected 'spearman' or 'pearson'")


def ic_summary(ic: np.ndarray) -> dict:
    """Summarize an IC series: level, stability, and whether it is
    distinguishable from zero.

    ``ir`` is the information ratio (mean / std) -- the stability measure
    that matters more than the raw mean, since a factor with a small but
    steady IC beats one with a large erratic IC. ``t_stat``/``p_value`` test
    the mean against zero, treating bars as independent; with overlapping
    horizons (``h > 1``) they are optimistic, because consecutive bars share
    most of their return window.
    """
    ic = np.asarray(ic, dtype='float64')
    valid = ic[~np.isnan(ic)]
    n_obs = int(valid.size)
    if n_obs == 0:
        return {'n_obs': 0, 'mean': float('nan'), 'std': float('nan'), 'ir': float('nan'),
                't_stat': float('nan'), 'p_value': float('nan'), 'positive_rate': float('nan')}
    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if n_obs > 1 else float('nan')
    ir = mean / std if (not np.isnan(std) and std > 0) else float('nan')
    if n_obs > 1 and not np.isnan(std) and std > 0:
        t_stat = mean / (std / np.sqrt(n_obs))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n_obs - 1))
    else:
        t_stat = float('nan')
        p_value = float('nan')
    return {
        'n_obs': n_obs, 'mean': mean, 'std': std, 'ir': ir,
        't_stat': float(t_stat), 'p_value': p_value,
        'positive_rate': float((valid > 0).mean()),
    }


def ic_decay(panel: FactorPanel, fwds: Dict[int, np.ndarray], method: str = 'spearman', min_names: int = 3) -> dict:
    """``{horizon: ic_summary}`` -- how fast the signal's edge fades.

    A factor whose IC holds up out to 20 bars can be rebalanced slowly and
    cheaply; one whose IC is gone by bar 5 has to be traded fast enough that
    commission and slippage may eat it. Compare against
    :func:`factor_autocorr`: if the IC decays much faster than the factor
    itself does, the signal is going stale rather than the position.
    """
    return {
        h: ic_summary(ic_series(panel, fwd, method=method, min_names=min_names))
        for h, fwd in sorted(fwds.items())
    }


def _group_labels(values: np.ndarray, n_groups: int) -> np.ndarray:
    """``float64`` group index in ``[0, n_groups - 1]`` per cell, NaN where no score.

    Group 0 is the weakest end of each bar's cross-section, ``n_groups - 1``
    the strongest.
    """
    ranks = cs_rank(values)
    with np.errstate(invalid='ignore'):
        labels = np.floor(ranks * n_groups)
        labels = np.clip(labels, 0, n_groups - 1)
    return np.where(np.isnan(ranks), np.nan, labels)


def _return_summary(series: np.ndarray, horizon: int) -> dict:
    """Mean / t-stat / annualized Sharpe of a per-bar return series.

    Each observation spans ``horizon`` bars, so ``horizon`` new bars elapse
    per independent observation -- annualization divides the year's bars by
    that, and overlapping windows (``horizon > 1``) still leave the t-stat
    optimistic since neighbouring observations share most of their window.
    """
    series = np.asarray(series, dtype='float64')
    valid = series[~np.isnan(series)]
    n_obs = int(valid.size)
    if n_obs == 0:
        return {'n_obs': 0, 'mean': float('nan'), 'std': float('nan'), 't_stat': float('nan'), 'sharpe': float('nan')}
    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if n_obs > 1 else float('nan')
    if n_obs > 1 and not np.isnan(std) and std > 0:
        t_stat = mean / (std / np.sqrt(n_obs))
        periods_per_year = TRADING_DAYS_PER_YEAR / max(int(horizon), 1)
        sharpe = mean / std * np.sqrt(periods_per_year)
    else:
        t_stat = float('nan')
        sharpe = float('nan')
    return {'n_obs': n_obs, 'mean': mean, 'std': std, 't_stat': float(t_stat), 'sharpe': float(sharpe)}


def quantile_returns(panel: FactorPanel, fwd: np.ndarray, n_groups: int, horizon: int) -> dict:
    """Sort each bar's cross-section into ``n_groups`` buckets and measure
    what each earned over the following ``horizon`` bars.

    Returns ``group_returns`` (``float64[n_bars, n_groups]``, the mean
    forward return of each bucket on each bar), the per-group summary, the
    long-short spread (top bucket minus bottom), and ``monotonicity`` -- the
    rank correlation between bucket index and mean bucket return.
    Monotonicity is the check that matters most: a factor whose extreme
    buckets separate but whose middle is scrambled is usually picking up an
    outlier effect, not a monotone relationship, and will not survive a
    different universe.

    With a small universe this is coarse by construction: 7 products across
    5 buckets is 1-2 products each, so a bucket's "mean" is often a single
    product. A warning is logged when average coverage cannot fill the buckets.
    """
    if n_groups < 2:
        raise ValueError(f"quantile_returns: n_groups must be >= 2, got {n_groups}")
    _check_aligned(panel, fwd)

    labels = _group_labels(panel.values, n_groups)
    n_bars = panel.n_bars
    group_returns = np.full((n_bars, n_groups), np.nan, dtype='float64')
    for g in range(n_groups):
        with np.errstate(invalid='ignore'):
            mask = (labels == g) & np.isfinite(fwd)
        counts = mask.sum(axis=1)
        sums = np.where(mask, fwd, 0.0).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            group_returns[:, g] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    groups = {g: _return_summary(group_returns[:, g], horizon) for g in range(n_groups)}

    spread_series = group_returns[:, -1] - group_returns[:, 0]
    spread = _return_summary(spread_series, horizon)

    with np.errstate(invalid='ignore'):
        mean_returns = np.array(
            [np.nanmean(group_returns[:, g]) if np.isfinite(group_returns[:, g]).any() else np.nan
             for g in range(n_groups)]
        )
    bucket_idx = np.arange(n_groups, dtype='float64')
    valid_g = ~np.isnan(mean_returns)
    if valid_g.sum() >= 2:
        monotonicity = float(stats.spearmanr(bucket_idx[valid_g], mean_returns[valid_g]).statistic)
    else:
        monotonicity = float('nan')

    coverage = panel.coverage()
    avg_coverage = float(coverage.mean()) if coverage.size else 0.0
    if avg_coverage < n_groups:
        logger.warning(
            "quantile_returns: average coverage (%.1f) is below n_groups=%d -- "
            "buckets will often hold a single product or none.", avg_coverage, n_groups,
        )

    return {
        'n_groups': n_groups, 'horizon': horizon,
        'group_returns': group_returns, 'groups': groups,
        'spread_series': spread_series, 'spread': spread,
        'monotonicity': monotonicity,
    }


def cumulative_curves(group_returns: np.ndarray, horizon: int):
    """Compound each bucket's forward returns into an equity curve starting at 1.0.

    Returns ``(rows, curves)``: ``rows`` is the bar index behind each point of
    ``curves``, which is ``float64[len(rows), n_groups]``.

    Consecutive rows of a ``horizon > 1`` matrix overlap -- the observation on
    bar ``t`` and the one on bar ``t + 1`` share all but one bar of the same
    price move -- so compounding every row counts each move ``horizon`` times
    over and inflates the curve exponentially rather than proportionally.
    Only every ``horizon``-th observation is independent, so only those are
    sampled; ``rows`` reports which bars they were, and ``curves[k]`` is the
    cumulative value *before* that sample's own return is realized -- the
    return recorded at ``rows[k]`` shows up in ``curves[k + 1]``, which is
    what "starting at 1.0" means for the first point.
    """
    group_returns = np.asarray(group_returns, dtype='float64')
    n_bars, n_groups = group_returns.shape
    horizon = max(int(horizon), 1)
    rows = np.arange(0, n_bars, horizon, dtype='int64')
    if rows.size == 0:
        return rows, np.zeros((0, n_groups), dtype='float64')
    sampled = np.nan_to_num(group_returns[rows], nan=0.0)
    growth = np.ones((rows.size, n_groups), dtype='float64')
    if rows.size > 1:
        growth[1:] = 1.0 + sampled[:-1]
    curves = np.cumprod(growth, axis=0)
    return rows, curves


def factor_turnover(panel: FactorPanel, n_groups: int = 3, rebalance: int = 5) -> dict:
    """How much of the extreme buckets' membership changes per rebalance.

    Reported as the fraction of the top (and bottom) bucket replaced between
    consecutive rebalances, so 0.0 is a bucket that never changes hands and
    1.0 one that turns over completely. This is the cost side of the
    picture: a factor with a strong IC but near-total turnover pays
    commission and slippage on every name, every rebalance.
    """
    labels = _group_labels(panel.values, n_groups)
    n_bars = panel.n_bars
    rebalance = max(int(rebalance), 1)
    sample_idx = range(0, n_bars, rebalance)
    buckets = {'top': n_groups - 1, 'bottom': 0}

    out = {}
    for name, g in buckets.items():
        fracs: List[float] = []
        prev = None
        for i in sample_idx:
            members = set(np.flatnonzero(labels[i] == g).tolist())
            if prev is not None and len(prev) > 0:
                fracs.append(len(prev - members) / len(prev))
            prev = members
        out[name] = float(np.mean(fracs)) if fracs else float('nan')
    return out


def factor_autocorr(panel: FactorPanel, lags: Iterable[int] = (1, 5, 10, 20)) -> Dict[int, float]:
    """``{lag: mean cross-sectional rank autocorrelation}`` of the factor itself.

    How long a score persists, independent of whether it predicts anything.
    Read alongside :func:`ic_decay`: a factor that stays put for 20 bars can
    be rebalanced every 20 bars regardless of how its IC behaves, while one
    that reshuffles daily forces a fast rebalance and the costs that come with it.
    """
    values = panel.values
    n_bars = values.shape[0]
    out: Dict[int, float] = {}
    for lag_ in lags:
        lag_ = int(lag_)
        shifted = np.full(values.shape, np.nan, dtype='float64')
        if lag_ < n_bars:
            shifted[lag_:] = values[:n_bars - lag_]
        ra, rb = joint_ranks(values, shifted)
        corr = rowwise_corr(ra, rb, min_names=3)
        valid = corr[~np.isnan(corr)]
        out[lag_] = float(valid.mean()) if valid.size else float('nan')
    return out


def coverage_summary(panel: FactorPanel) -> dict:
    """How many products actually carry a score, over time.

    A factor whose average coverage is well below the loaded universe is
    being thinned by its own warmup or by NaNs it produces, and every
    statistic above is then describing a smaller cross-section than the
    user thinks.
    """
    coverage = panel.coverage()
    n_bars = panel.n_bars
    n_symbols = panel.n_symbols
    nonzero = np.flatnonzero(coverage > 0)
    return {
        'n_symbols': n_symbols,
        'n_bars': n_bars,
        'mean': float(coverage.mean()) if n_bars else float('nan'),
        'min': int(coverage.min()) if n_bars else 0,
        'max': int(coverage.max()) if n_bars else 0,
        'first_scored_bar': int(nonzero[0]) if nonzero.size else -1,
        'bars_with_full_coverage': int((coverage == n_symbols).sum()),
    }


def annual_slices(
    panel: FactorPanel,
    fwd: np.ndarray,
    dates,
    horizon: int = 1,
    n_groups: int = 3,
    method: str = 'spearman',
    min_names: int = 3,
) -> Dict[int, dict]:
    """Per-calendar-year IC and long-short return -- the number
    :func:`ic_summary`/:func:`quantile_returns` compute over the full sample
    can hide exactly this kind of reversal (a strategy's full-sample Sharpe
    of 0.9 has been observed to hide a swing from +2.94 to -1.93 across
    validation windows on this codebase's own ``momentum_barrier`` strategy).
    A factor's aggregate IC should never be reported without this table next
    to it: a single sign flip across years is a reason to check for a
    structural break before trusting the aggregate number.
    """
    _check_aligned(panel, fwd)
    dates = pd.DatetimeIndex(np.asarray(dates))
    years = dates.year.to_numpy()

    ic = ic_series(panel, fwd, method=method, min_names=min_names)
    labels = _group_labels(panel.values, n_groups)

    out: Dict[int, dict] = {}
    for year in sorted(set(int(y) for y in years)):
        mask = years == year
        year_ic = ic[mask]
        ic_stats = ic_summary(year_ic)

        year_fwd = fwd[mask]
        year_labels = labels[mask]
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            top = np.where(year_labels == n_groups - 1, year_fwd, np.nan)
            bottom = np.where(year_labels == 0, year_fwd, np.nan)
            top_mean = np.nanmean(top, axis=1) if top.size else np.array([])
            bottom_mean = np.nanmean(bottom, axis=1) if bottom.size else np.array([])
        spread = _return_summary(top_mean - bottom_mean, horizon) if top_mean.size else _return_summary(
            np.array([]), horizon
        )

        out[year] = {
            'n_bars': int(mask.sum()),
            'ic_mean': ic_stats['mean'],
            'ic_ir': ic_stats['ir'],
            'ic_t_stat': ic_stats['t_stat'],
            'long_short_mean': spread['mean'],
            'long_short_sharpe': spread['sharpe'],
        }
    return out
