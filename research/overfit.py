"""Overfitting diagnostics for one already-chosen parameter set.

Five independent checks, driven by :mod:`research.validate`:

- ``is_oos_decay``: does the score in the earlier, longer stretch survive
  into the later, unseen one?
- ``pbo_cscv``: Probability of Backtest Overfitting via Combinatorially
  Symmetric Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu, 2015).
- ``deflated_sharpe_ratio``: probability the Sharpe is genuinely positive
  after correcting for how many configurations were tried and for
  non-normal returns (Bailey & Lopez de Prado, 2014).
- ``plateau_check``: does the score survive a one-step perturbation in each
  dimension, or is this a lone spike in the parameter surface?
- ``block_bootstrap``: how wide is the Sharpe's confidence interval once the
  return series is resampled in blocks -- i.e. how much of the backtest is a
  draw the sample size cannot distinguish from luck?

None of this is strategy-specific: every function takes plain arrays,
score lists, or a ``space``/``evaluate`` pair. Nothing here searches, and
nothing here returns a parameter set -- these answer "how likely is this to
be noise", never "what should I use instead".
"""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Callable

import numpy as np
from scipy import stats

from core.metrics import DEFAULT_RISK_FREE_RATE
from core.params import Categorical, Float, Int
from research.space import check_constraints

_EULER_GAMMA = 0.5772156649015329


def is_oos_decay(train_scores: list, valid_scores: list) -> dict:
    """Compare the mean in-sample against the mean out-of-sample fold score.

    With the parameters held fixed there is no search to overfit *here*, so
    what this measures is whether the edge holds up as the window moves
    forward: regime drift, plus whatever fitting happened before the
    parameters reached this tool. A ratio near 1 means the later folds look
    like the earlier ones; a collapse means the number in the headline
    backtest is carried by one stretch of history.
    """
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
    overfitting, computed on a ``[n_candidates, n_bars]`` matrix of daily
    returns, one row per candidate parameter set over the same span.

    Splits the time axis into ``n_blocks`` contiguous blocks. For every way
    of assigning half the blocks to an in-sample (IS) half (the rest are
    out-of-sample, OOS) -- ``C(n_blocks, n_blocks/2)`` assignments in
    total -- finds the candidate with the best IS performance and checks
    whether its OOS performance lands at or below the OOS median. PBO is
    the fraction of assignments where it does: how often would picking the
    apparent IS winner have handed you a coin-flip-or-worse OOS result.

    **The answer is only as wide as the candidate set.** Fed the one-step
    neighbourhood that :mod:`research.validate` builds, this reads as "if I
    picked among these nearby variants by in-sample performance, would the
    choice have held up" -- a narrower question than the same statistic
    computed over a real search's trials, and it should be quoted as such.
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


def deflated_sharpe_ratio(returns_matrix: np.ndarray, best_idx: int, *, n_trials: int = None) -> dict:
    """Probability that row ``best_idx``'s Sharpe is genuinely positive,
    deflated for the number of configurations tried (the multiple-testing
    correction) and for the skew/kurtosis of its own returns.

    ``n_trials`` defaults to the row count, which is right when the matrix
    *is* the set of things that were tried. It is an override because that
    is usually not the case here: :mod:`research.validate` builds a dozen-odd
    neighbours to estimate the spread of Sharpe across configurations, while
    the number of configurations actually tried is however many the author
    ran by hand before settling -- often far more. The correction belongs on
    that count, so ``ft.py validate --trials-tried`` passes it through.
    Under-declaring it inflates the result, which is the direction this whole
    module exists to push back on.
    """
    n_rows, n_obs = returns_matrix.shape
    n_trials = n_rows if n_trials is None else max(2, int(n_trials))
    if n_rows < 2 or n_obs < 3:
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
        'skew': skew, 'kurtosis': kurt, 'n_trials': n_trials,
        # How many rows the Sharpe *spread* was estimated from, which is a
        # different number from how many configurations were tried whenever
        # `n_trials` was declared. Reporting only the latter made the two
        # indistinguishable in the JSON.
        'n_candidates': n_rows, 'n_obs': n_obs,
    }


def _max_drawdown_rows(equity: np.ndarray) -> np.ndarray:
    """Per-row max drawdown as a fraction, on ``[n_rows, n_bars]`` equity.

    Mirrors ``core.metrics.compute_metrics``: once the running peak is
    non-positive the drawdown is undefined as a fraction of it, and that
    region counts as a full 100% drawdown rather than a division by zero.
    """
    running_max = np.maximum.accumulate(equity, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (running_max - equity) / running_max
    return np.where(running_max > 0, ratio, 1.0).max(axis=1)


def block_bootstrap(
    returns,
    *,
    n_draws: int = 2000,
    block: int = 20,
    periods_per_year: float = 244.0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    seed: int = 0,
) -> dict:
    """Resample ``returns`` in contiguous circular blocks and report how much
    of the backtest's Sharpe and drawdown the sample size can actually pin
    down.

    Blocks rather than individual days because daily returns are not
    independent -- a trend-following run's good and bad stretches cluster,
    and shuffling day by day would break exactly the autocorrelation that
    makes a Sharpe over 1500 bars less informative than 1500 independent
    draws would be. ``block`` should be at least as long as a typical
    holding period.

    Returns annualized Sharpe (point estimate, 95% interval, and the share
    of resamples that came out positive) and the max-drawdown distribution
    the observed drawdown is one draw from, in percent to match
    ``core.metrics``. A wide interval straddling zero is the finding: the
    equity curve is compatible with having no edge at all.

    The Sharpe is an *excess* ratio over ``risk_free_rate``, and the
    ``periods_per_year`` the caller passes should come from
    ``core.metrics.annualization_factor`` over the same dates. Both exist so
    the point estimate here lands on ``compute_metrics``' ``sharpe_ratio``
    exactly; leave either out and the report quotes two different Sharpes for
    one equity curve. Drawdown is measured on the raw returns, not the excess
    ones -- an account draws down in the money it actually holds.

    A series too short to resample is answered with NaN and
    ``insufficient=True`` -- the same "this check did not run" signal
    ``pbo_cscv`` gives a window it cannot split, which has to stay
    distinguishable from a check that ran and found nothing.
    """
    r = np.asarray(returns, dtype='float64').ravel()
    r = r[np.isfinite(r)]
    n_obs = int(r.size)
    block = max(1, int(block))

    nan = float('nan')
    if n_obs < 30 or n_obs < 2 * block:
        return {
            'insufficient': True, 'n_obs': n_obs, 'block': block, 'n_draws': 0,
            'periods_per_year': periods_per_year, 'risk_free_rate': risk_free_rate,
            'sharpe': {'point': nan, 'ci_low': nan, 'ci_high': nan, 'p_positive': nan},
            'max_drawdown': {'observed': nan, 'median': nan, 'p95': nan},
        }

    annualize = math.sqrt(periods_per_year)
    risk_free_per_period = risk_free_rate / periods_per_year

    def _sharpe(rows: np.ndarray) -> np.ndarray:
        excess = rows - risk_free_per_period
        mean, std = excess.mean(axis=1), excess.std(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(std > 1e-12, mean / std * annualize, 0.0)

    observed = r[None, :]
    point_sharpe = float(_sharpe(observed)[0])
    observed_dd = float(_max_drawdown_rows(np.cumprod(1.0 + observed, axis=1))[0])

    rng = np.random.default_rng(seed)
    n_blocks = -(-n_obs // block)   # ceil: enough blocks to cover the series
    offsets = np.arange(block)

    # Chunked so a large `n_draws` on a long series cannot balloon into a
    # multi-hundred-MB index array: the draws are independent, so the batch
    # size is free to be whatever keeps one chunk small.
    chunk = max(1, int(2_000_000 // max(1, n_blocks * block)))
    sharpes, drawdowns = [], []
    for start in range(0, n_draws, chunk):
        size = min(chunk, n_draws - start)
        starts = rng.integers(0, n_obs, size=(size, n_blocks))
        idx = (starts[:, :, None] + offsets[None, None, :]) % n_obs
        samples = r[idx.reshape(size, -1)[:, :n_obs]]
        sharpes.append(_sharpe(samples))
        drawdowns.append(_max_drawdown_rows(np.cumprod(1.0 + samples, axis=1)))

    sharpes = np.concatenate(sharpes)
    drawdowns = np.concatenate(drawdowns)

    return {
        'insufficient': False, 'n_obs': n_obs, 'block': block, 'n_draws': n_draws,
        'periods_per_year': periods_per_year, 'risk_free_rate': risk_free_rate,
        'sharpe': {
            'point': point_sharpe,
            'ci_low': float(np.percentile(sharpes, 2.5)),
            'ci_high': float(np.percentile(sharpes, 97.5)),
            'p_positive': float(np.mean(sharpes > 0.0)),
        },
        'max_drawdown': {
            'observed': observed_dd * 100.0,
            'median': float(np.median(drawdowns)) * 100.0,
            'p95': float(np.percentile(drawdowns, 95)) * 100.0,
        },
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


def plateau_check(strategy_cls: type, space: dict, params: dict, evaluate: Callable) -> dict:
    """For each dimension, perturb ``params`` one step in each direction
    (per ``space``'s bounds/step) and re-score via ``evaluate(params) ->
    float``. A flat neighbourhood means the edge does not depend on the exact
    number; a >30% one-step drop means these particular values are load-
    bearing, which on a few years of daily bars is what fitting noise looks
    like.
    """
    base_score = evaluate(params)
    # `abs(base_score) > 1e-9`, not `base_score > 1e-9`. Gating on the *sign*
    # meant a parameter set whose base score was negative -- a losing strategy,
    # and precisely when this check earns its keep -- left
    # `drops` empty for every dimension, so `max_drop` fell to 0.0 and the
    # report came back reading "flat neighbourhood, no spikes" for a check
    # that never ran. A neighbour at -4.0 against a base of -1.5 is still a
    # collapse; dividing by the magnitude says so, and a neighbour that scores
    # *better* still yields a negative drop and flags nothing.
    measurable = abs(base_score) > 1e-9
    results = {}
    for name, spec in space.items():
        if name not in params:
            continue
        drops = []
        for neighbor in _neighbors(spec, params[name]):
            candidate = {**params, name: neighbor}
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
