"""One comparable number per window: strategy-agnostic, works purely off
``compute_metrics()``'s output dict.

``core.metrics.compute_metrics`` can return ``calmar_ratio`` /
``profit_factor`` / ``profit_loss_ratio`` as ``float('inf')`` (see
``core/metrics.py``), and an empty window returns ``{}`` entirely. Neither
can be compared or averaged raw, so all clamping happens here, in the
research layer, rather than in ``core``.
"""

from __future__ import annotations

import math


def _finite(x) -> float:
    if x is None:
        return 0.0
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def window_years(n_bars: int, trading_days_per_year: float = 244.0) -> float:
    """Approximate calendar years spanned by ``n_bars`` trading days."""
    return max(n_bars, 1) / trading_days_per_year


DEFAULT_SPARSE_PENALTY = 0.5


def score(
    metrics: dict,
    *,
    window_years: float,
    min_trades_per_year: float = 4.0,
    dd_cap: float = 0.35,
    sparse_penalty: float = DEFAULT_SPARSE_PENALTY,
) -> float:
    """Single-window score: Sharpe, discounted for how little of the window's
    expected trade count the run actually produced, then penalized for excess
    drawdown beyond ``dd_cap`` and for forced liquidations. Used wherever
    windows have to be compared on one axis -- fold against fold, and a
    parameter set against its neighbours.

    Why not quote Sharpe directly: a Sharpe of 3 off two trades is not a
    better window than a Sharpe of 1 off forty, it is a window with almost no
    evidence in it, and a short validation fold produces those constantly.

    **The one invariant: trading less can never raise a score.** Scaling a
    score toward zero *raises* a negative number, so a single multiplicative
    coverage factor would rank a parameter set that lost money sparsely above
    one that lost less but traded often enough to be measured -- and would
    score a set that never traded at all as exactly ``0.0``, better than every
    honest loser. Two adjustments hold the invariant, deliberately asymmetric:

    * A **positive** Sharpe is discounted by ``coverage``, because the window
      cannot support it.
    * A **negative** one is taken at face value, no discount. Explaining a bad
      sample away as "too few trades to judge" is exactly the move that lets a
      backtest flatter itself, which is the same reading this codebase takes
      everywhere else a daily bar leaves a choice open.

    On top of that, ``sparse_penalty * (1 - coverage)`` marks down whatever is
    missing regardless of sign, which puts a floor of ``-sparse_penalty``
    under a window that did nothing at all.
    """
    if not metrics:
        return -10.0

    need = max(min_trades_per_year * window_years, 1.0)
    coverage = min(1.0, metrics.get('n_trades', 0) / need)

    sharpe = _finite(metrics.get('sharpe_ratio', 0.0))
    s = sharpe * coverage if sharpe > 0 else sharpe
    s -= sparse_penalty * (1.0 - coverage)

    dd = _finite(metrics.get('max_drawdown', 0.0)) / 100.0
    if dd > dd_cap:
        s -= 2.0 * (dd - dd_cap)

    s -= 1.0 * metrics.get('n_forced_liquidations', 0)

    return float(max(-10.0, min(10.0, s)))
