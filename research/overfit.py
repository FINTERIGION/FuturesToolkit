"""Post-hoc overfitting diagnostics for an Optuna study.

Four independent checks, all computed once after the search finishes:

- ``is_oos_decay``: does the best trial's in-sample score collapse out of sample?
- ``pbo_cscv``: Probability of Backtest Overfitting via Combinatorially
  Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu, 2015).
- ``deflated_sharpe_ratio``: probability the best trial's Sharpe is
  genuinely positive after correcting for how many configurations were
  tried and for non-normal returns (Bailey & Lopez de Prado, 2014).
- ``plateau_check``: does the score survive a one-step perturbation in each
  tuned dimension, or is the optimum a lone spike?

None of this is strategy-specific: every function takes plain arrays,
score lists, or a ``space``/``evaluate`` pair.
"""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Callable

import numpy as np
from scipy import stats

from core.params import Categorical, Float, Int
from research.space import check_constraints

_EULER_GAMMA = 0.5772156649015329


def is_oos_decay(train_scores: list, valid_scores: list) -> dict:
    """Compare a trial's mean in-sample vs. out-of-sample fold score."""
    is_score = statistics.fmean(train_scores) if train_scores else 0.0
    oos_score = statistics.fmean(valid_scores) if valid_scores else 0.0
    if abs(is_score) > 1e-9:
        ratio = oos_score / is_score
    else:
        ratio = float('nan')
    warn = is_score > 0 and (math.isnan(ratio) or ratio < 0.5)
    return {'is_score': is_score, 'oos_score': oos_score, 'ratio': ratio, 'warn': warn}


def _period_sharpe(returns: np.ndarray) -> np.ndarray:
    """Per-row (non-annualized) mean/std Sharpe, 0 where std is ~0."""
    mean = returns.mean(axis=1)
    std = returns.std(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(std > 1e-12, mean / std, 0.0)
    return out


def pbo_cscv(returns_matrix: np.ndarray, n_blocks: int = 16) -> dict:
    """Combinatorially Symmetric Cross-Validation probability of backtest
    overfitting, computed on a ``[n_trials, n_bars]`` matrix of per-trial
    daily returns over the full pre-holdout span.

    Splits the time axis into ``n_blocks`` contiguous blocks. For every way
    of assigning half the blocks to an in-sample (IS) half (the rest are
    out-of-sample, OOS) -- ``C(n_blocks, n_blocks/2)`` assignments in
    total -- finds the trial with the best IS performance and checks
    whether its OOS performance lands at or below the OOS median. PBO is
    the fraction of assignments where it does: how often would picking the
    apparent IS winner have handed you a coin-flip-or-worse OOS trial.
    """
    n_trials, n_bars = returns_matrix.shape
    if n_trials < 2 or n_bars < 2:
        return {'pbo': float('nan'), 'n_combinations': 0, 'n_blocks': 0}

    # Clamp first, *then* force an even count. The other order ran the parity
    # adjustment inside the `min`, so clamping to a short `n_bars` could hand
    # back an odd number again -- 15 bars gave 15 blocks, splitting 7
    # in-sample against 8 out-of-sample. CSCV's whole premise is that the two
    # halves are interchangeable, which an asymmetric split quietly breaks.
    n_blocks = max(2, min(n_blocks, n_bars))
    n_blocks -= n_blocks % 2
    if n_blocks < 2:
        return {'pbo': float('nan'), 'n_combinations': 0, 'n_blocks': n_blocks}

    edges = np.linspace(0, n_bars, n_blocks + 1, dtype=int)
    blocks = [returns_matrix[:, edges[i]:edges[i + 1]] for i in range(n_blocks)]
    half = n_blocks // 2

    below = total = 0
    for is_idx in itertools.combinations(range(n_blocks), half):
        oos_idx = [i for i in range(n_blocks) if i not in is_idx]
        is_sharpe = _period_sharpe(np.concatenate([blocks[i] for i in is_idx], axis=1))
        oos_sharpe = _period_sharpe(np.concatenate([blocks[i] for i in oos_idx], axis=1))
        best = int(np.argmax(is_sharpe))
        rank = int(np.sum(oos_sharpe <= oos_sharpe[best]))  # 1..n_trials
        omega = rank / (n_trials + 1)
        total += 1
        if omega <= 0.5:
            below += 1

    return {'pbo': below / total, 'n_combinations': total, 'n_blocks': n_blocks}


def deflated_sharpe_ratio(returns_matrix: np.ndarray, best_idx: int) -> dict:
    """Probability the best trial's Sharpe is genuinely positive, deflated
    for the number of configurations tried (``n_trials``, the multiple-
    testing correction) and for the skew/kurtosis of its own returns.
    """
    n_trials, n_obs = returns_matrix.shape
    if n_trials < 2 or n_obs < 3:
        return {'dsr': float('nan'), 'sr0': float('nan'), 'sr_hat': float('nan')}

    per_trial_sharpe = _period_sharpe(returns_matrix)
    sr_hat = float(per_trial_sharpe[best_idx])

    var_sr = float(np.var(per_trial_sharpe, ddof=1))
    sr0 = 0.0
    if var_sr > 0:
        sr0 = math.sqrt(var_sr) * (
            (1 - _EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / n_trials)
            + _EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        )

    best_returns = returns_matrix[best_idx]
    skew = float(stats.skew(best_returns))
    kurt = float(stats.kurtosis(best_returns, fisher=False))  # normal == 3

    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2))
    z = (sr_hat - sr0) * math.sqrt(n_obs - 1) / denom
    dsr = float(stats.norm.cdf(z))

    return {
        'dsr': dsr, 'sr0': sr0, 'sr_hat': sr_hat, 'z': z,
        'skew': skew, 'kurtosis': kurt, 'n_trials': n_trials, 'n_obs': n_obs,
    }


def _neighbors(spec, value):
    if isinstance(spec, Int):
        step = max(1, spec.step)
        return [v for v in (value - step, value + step) if spec.low <= v <= spec.high]
    if isinstance(spec, Float):
        step = spec.step if spec.step else (spec.high - spec.low) * 0.05
        return [v for v in (value - step, value + step) if spec.low <= v <= spec.high]
    if isinstance(spec, Categorical):
        return [c for c in spec.choices if c != value][:2]
    return []


def plateau_check(strategy_cls: type, space: dict, best_params: dict, evaluate: Callable) -> dict:
    """For each tuned dimension, perturb ``best_params`` one step in each
    direction (per ``space``'s bounds/step) and re-score via
    ``evaluate(params) -> float``. A flat neighborhood is a trustworthy
    optimum; a >30% one-step score drop usually means the search landed on
    noise rather than a real, robust edge.
    """
    base_score = evaluate(best_params)
    # `abs(base_score) > 1e-9`, not `base_score > 1e-9`. Gating on the *sign*
    # meant a search whose best score was negative -- the normal outcome for a
    # space with no edge, and precisely when this check earns its keep -- left
    # `drops` empty for every dimension, so `max_drop` fell to 0.0 and the
    # report came back reading "flat neighbourhood, no spikes" for a check
    # that never ran. A neighbour at -4.0 against a base of -1.5 is still a
    # collapse; dividing by the magnitude says so, and a neighbour that scores
    # *better* still yields a negative drop and flags nothing.
    measurable = abs(base_score) > 1e-9
    results = {}
    for name, spec in space.items():
        if name not in best_params:
            continue
        drops = []
        for neighbor in _neighbors(spec, best_params[name]):
            candidate = {**best_params, name: neighbor}
            if not check_constraints(strategy_cls, candidate):
                continue
            neighbor_score = evaluate(candidate)
            if measurable:
                drops.append((base_score - neighbor_score) / abs(base_score))
        max_drop = max(drops) if drops else 0.0
        results[name] = {
            'max_drop_pct': max_drop * 100.0,
            'flags_spike': bool(drops) and max_drop > 0.30,
            # A base score of ~0 leaves no scale to measure a relative drop
            # against, and every neighbour may be ruled out by the strategy's
            # own constraints. Either way the answer is "not checked", which
            # has to be distinguishable from "checked and flat".
            'evaluated': bool(drops),
        }
    return {'base_score': base_score, 'dimensions': results}
