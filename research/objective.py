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
    drawdown beyond ``dd_cap`` and for forced liquidations. Used to score every
    fold and the holdout run alike.

    **The one invariant: trading less can never raise a score.** This used to
    be a single multiplicative factor, and scaling toward zero *raises* a
    negative number -- so a parameter set that lost money sparsely outranked
    one that lost less but traded often enough to be measured, and a set that
    never traded at all scored exactly ``0.0``, beating every honest loser in
    the space. With no positive-Sharpe configuration anywhere in a search, the
    global optimum of this objective was to do nothing, handed back as
    ``best_params`` with a clean-looking value near zero. A search over a
    space that does not work should say so with a negative number.

    Two adjustments hold that invariant, and they are deliberately asymmetric:

    * A **positive** Sharpe is discounted by ``coverage``, because a Sharpe of
      3 off two trades is a number the window cannot support.
    * A **negative** one is taken at face value, no discount. Explaining a bad
      sample away as "too few trades to judge" is exactly the move that lets a
      backtest flatter itself, which is the same reading this codebase takes
      everywhere else a daily bar leaves a choice open.

    On top of that, ``sparse_penalty * (1 - coverage)`` marks down whatever is
    missing regardless of sign, which is what puts a floor of
    ``-sparse_penalty`` under the do-nothing corner of the space.

    ``sparse_penalty`` is the dial. At the default 0.5 a promising-but-thin
    configuration keeps a positive score, so the sampler goes on exploring
    around it; raising it toward 1.0 demands the evidence be there before a
    configuration counts at all.
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
