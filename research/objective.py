"""Objective scoring for Optuna trials: strategy-agnostic, works purely off
``compute_metrics()``'s output dict.

``core.metrics.compute_metrics`` can return ``calmar_ratio`` /
``profit_factor`` / ``profit_loss_ratio`` as ``float('inf')`` (see
``core/metrics.py``), and an empty window returns ``{}`` entirely. Both
would poison Optuna's TPE sampler if handed to it raw, so all clamping
happens here, in the research layer, rather than in ``core``.
"""

from __future__ import annotations

import math
import statistics


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


def score(
    metrics: dict,
    *,
    window_years: float,
    min_trades_per_year: float = 4.0,
    dd_cap: float = 0.35,
) -> float:
    """Single-window score: Sharpe, penalized for too few trades for the
    window's length, excess drawdown beyond ``dd_cap``, and forced
    liquidations. Used to score every fold and the holdout run alike.
    """
    if not metrics:
        return -10.0

    s = _finite(metrics.get('sharpe_ratio', 0.0))

    need = max(min_trades_per_year * window_years, 1.0)
    n_trades = metrics.get('n_trades', 0)
    if n_trades < need:
        s *= n_trades / need

    dd = _finite(metrics.get('max_drawdown', 0.0)) / 100.0
    if dd > dd_cap:
        s -= 2.0 * (dd - dd_cap)

    s -= 1.0 * metrics.get('n_forced_liquidations', 0)

    return float(max(-10.0, min(10.0, s)))


def fold_objective(fold_scores: list, *, lambda_std: float = 0.5) -> float:
    """Aggregate a trial's per-fold valid scores into one Optuna objective:
    the mean, penalized by cross-fold standard deviation. Rewards params
    that hold up across every fold over params that spike in just one.
    """
    if not fold_scores:
        return -10.0
    mean = statistics.fmean(fold_scores)
    std = statistics.pstdev(fold_scores) if len(fold_scores) > 1 else 0.0
    return mean - lambda_std * std
