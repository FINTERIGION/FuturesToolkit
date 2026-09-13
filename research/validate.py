"""Overfitting checks for one already-chosen parameter set.

``validate_strategy`` is the only entry point ``ft.py validate`` needs. It
takes a strategy and the parameters you intend to trade, and answers one
question from five directions: **how much of this backtest is the strategy,
and how much is this particular slice of history?**

It never searches, and it never returns a parameter set. That is the point.
The tool it replaces ran an Optuna study over anchored walk-forward folds and
handed back a winner, which on a handful of correlated products and a few
years of daily bars mostly found the luckiest corner of the space rather than
an edge. The diagnostics it computed afterwards were the useful half, so they
are what survived -- rewired to run against a fixed parameter set.

The five checks:

1. **Walk-forward consistency** -- the same parameters scored fold by fold,
   plus ``is_oos_decay`` over the anchored train windows against their paired
   validation windows.
2. **Sub-period stability** -- each calendar year (or each of ``n_periods``
   equal stretches) run in isolation. A full-sample Sharpe of 0.9 can hide a
   +2.9 year and a -1.9 year, and the average is not what gets traded.
3. **Parameter sensitivity** -- ``plateau_check`` over a one-step
   neighbourhood in every dimension. If the result needs exactly these
   numbers, the numbers came from the noise.
4. **Block bootstrap** -- a confidence interval on the Sharpe and a
   distribution for the max drawdown.
5. **PBO and Deflated Sharpe** -- computed on the neighbourhood grid from
   check 3, with the caveats spelled out in ``research.overfit``.

Checks 3 and 5 share one set of backtests: the neighbourhood is run once,
memoized, and read by both.
"""

from __future__ import annotations

import datetime
import json
import logging
import os

import numpy as np

from core.metrics import annualization_factor
from strategies import name_for
from research.objective import score, window_years
from research.overfit import (
    _neighbors, block_bootstrap, deflated_sharpe_ratio, is_oos_decay, pbo_cscv, plateau_check,
)
from research.runner_api import load_market, run_window
from research.space import check_constraints, resolve_space, spec_to_json
from research.splits import Window, anchored_walk_forward
from research.warmup import probe_warmup

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'validation')

CHECKS = ('walkforward', 'subperiod', 'sensitivity', 'bootstrap', 'pbo', 'dsr')

# A sub-period shorter than this is not a period, it is a remainder. Folded
# into its neighbour rather than reported as a year with four trades in it.
_MIN_PERIOD_BARS = 120


# ---------------------------------------------------------------------
# neighbourhood grid
# ---------------------------------------------------------------------

def _params_key(params: dict) -> tuple:
    """Hashable identity for a parameter set, for memoizing its backtest."""
    return tuple(sorted((k, repr(v)) for k, v in params.items()))


def build_grid(strategy_cls: type, space: dict, params: dict) -> list:
    """``params`` first, then every one-step neighbour in every dimension
    that the strategy's own ``constraints`` accept.

    Order matters only in that the base set is row 0 of the matrix fed to
    ``deflated_sharpe_ratio``; duplicates are dropped so a dimension whose
    neighbours collide with an existing entry does not weight it twice.
    """
    grid, seen = [dict(params)], {_params_key(params)}
    for name, spec in space.items():
        if name not in params:
            continue
        for neighbor in _neighbors(spec, params[name]):
            candidate = {**params, name: neighbor}
            key = _params_key(candidate)
            if key in seen or not check_constraints(strategy_cls, candidate):
                continue
            seen.add(key)
            grid.append(candidate)
    return grid


def _reserve_bars(market, strategy_cls: type, grid: list, margin: float = 1.2) -> int:
    """Leading bars every window reserves for indicator history: the slowest
    warmup anywhere in the grid, inflated by ``margin``.

    Exact rather than sampled. The search this replaces probed random
    configurations out of a continuous space and had to guess; a fixed
    neighbourhood is small and enumerable, so every member can be measured.
    """
    warmups = [0]
    for candidate in grid:
        warmups.append(probe_warmup(market, strategy_cls, candidate))
    return int(max(warmups) * margin) + 1


# ---------------------------------------------------------------------
# windowed runs
# ---------------------------------------------------------------------

def _window_row(name: str, window: Window, out: dict, **extra) -> dict:
    """One window's metrics, flattened to the handful worth comparing."""
    metrics = out['metrics']
    return {
        'name': name,
        'start': window.start,
        'end': window.end,
        'n_bars': window.n_bars,
        'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
        'total_return': metrics.get('total_return', 0.0),
        'max_drawdown': metrics.get('max_drawdown', 0.0),
        'n_trades': metrics.get('n_trades', 0),
        'blown_up': bool(metrics.get('blown_up')),
        'score': score(metrics, window_years=window_years(window.n_bars)),
        **extra,
    }


def _returns_over(window: Window, out: dict) -> np.ndarray:
    """A run's daily returns laid out over ``window``, zero where it did not
    trade.

    A run can come up short at either end, for opposite reasons: short at the
    *head* means the indicators needed more pad than they got, short at the
    *tail* means the account blew up and the run stopped there.
    Zero-filling the head unconditionally would shift a blown-up run's whole
    series forward in time -- and ``pbo_cscv`` compares rows block by block
    along exactly that axis, so a shifted row is scored against the wrong
    period in every single split. ``effective_start`` is absolute, in the
    market's own bar numbering, so the offset needs no guessing.
    """
    returns = np.zeros(window.n_bars, dtype='float64')
    records = out['result']['equity_records']
    if records:
        offset = max(0, out['effective_start'] - window.start)
        values = np.array([r['daily_return'] for r in records], dtype='float64')
        values = values[: window.n_bars - offset]
        returns[offset: offset + len(values)] = values
    return returns


def _calendar_periods(market, start: int, end: int) -> list:
    """``(label, Window)`` per calendar year over ``[start, end)``."""
    if end - start < 2:
        return [('all', Window('period_all', start, end))]
    years = market.dates[start:end].astype('datetime64[Y]').astype(int) + 1970
    bounds = np.flatnonzero(np.diff(years)) + 1
    edges = [0, *bounds.tolist(), end - start]
    return [
        (str(years[edges[i]]), Window(f'period_{years[edges[i]]}', start + edges[i], start + edges[i + 1]))
        for i in range(len(edges) - 1)
    ]


def _equal_periods(start: int, end: int, n_periods: int) -> list:
    edges = np.linspace(start, end, n_periods + 1, dtype=int)
    return [
        (f'{i + 1}/{n_periods}', Window(f'period_{i + 1}', int(edges[i]), int(edges[i + 1])))
        for i in range(n_periods)
    ]


def _merge_short_periods(periods: list) -> list:
    """Fold any stretch below ``_MIN_PERIOD_BARS`` into its neighbour.

    A partial first or last year is the usual case -- ``--start 2020-06-01``
    leaves half a 2020 -- and reporting it as its own row invites reading a
    Sharpe off twenty trading days. Merging keeps every bar accounted for.
    """
    if len(periods) < 2:
        return periods
    merged = []
    for label, window in periods:
        if merged and window.n_bars < _MIN_PERIOD_BARS:
            prev_label, prev = merged[-1]
            merged[-1] = (f'{prev_label}+{label}', Window(prev.name, prev.start, window.end))
        else:
            merged.append((label, window))
    # The *first* period can only merge forward, which the loop above cannot
    # do -- it has nothing behind it yet.
    if len(merged) > 1 and merged[0][1].n_bars < _MIN_PERIOD_BARS:
        (first_label, first), (second_label, second) = merged[0], merged[1]
        merged[:2] = [(f'{first_label}+{second_label}', Window(first.name, first.start, second.end))]
    return merged


# ---------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------

def _check_walkforward(run, folds: list) -> dict:
    train_rows, valid_rows = [], []
    for train_w, valid_w in folds:
        train_rows.append(_window_row(train_w.name, train_w, run(train_w)))
        valid_rows.append(_window_row(valid_w.name, valid_w, run(valid_w)))

    decay = is_oos_decay([r['score'] for r in train_rows], [r['score'] for r in valid_rows])
    return {
        'train': train_rows,
        'valid': valid_rows,
        'decay': decay,
        'warn': bool(decay['warn']),
        # Said here rather than left to the reader: with the parameters fixed,
        # "in sample" is only the earlier and longer stretch. A collapse means
        # the edge did not survive the window moving forward, which is regime
        # drift plus whatever fitting happened before this tool saw the
        # parameters -- not evidence about a search that did not happen.
        'note': 'Parameters are fixed, so IS/OOS here contrasts earlier, longer '
                'windows with later ones: it measures regime drift and any '
                'fitting done before validation, not selection inside this run.',
    }


def _check_subperiod(run, market, start: int, end: int, n_periods: int) -> dict:
    periods = _equal_periods(start, end, n_periods) if n_periods else _calendar_periods(market, start, end)
    periods = _merge_short_periods(periods)

    rows = [_window_row(label, window, run(window)) for label, window in periods]
    sharpes = [r['sharpe_ratio'] for r in rows]
    positive = sum(1 for s in sharpes if s > 0)
    worst = min(rows, key=lambda r: r['sharpe_ratio']) if rows else None
    fraction = positive / len(rows) if rows else float('nan')

    return {
        'periods': rows,
        'worst_period': worst,
        'positive_fraction': fraction,
        'n_periods': len(rows),
        # Both conditions, not either: one losing year in a strategy that wins
        # most of them is a drawdown, not a finding. A losing worst period
        # *and* a coin flip across the rest is a strategy carried by one
        # stretch of history.
        'warn': bool(rows) and worst['sharpe_ratio'] < 0 and fraction < 0.5,
    }


def _check_sensitivity(strategy_cls: type, space: dict, params: dict, evaluate) -> dict:
    plateau = plateau_check(strategy_cls, space, params, evaluate)
    dims = plateau['dimensions']
    return {
        **plateau,
        'spikes': sorted(k for k, v in dims.items() if v.get('flags_spike')),
        # A dimension whose neighbours were all ruled out by the strategy's
        # own constraints -- or a base score of ~0, which leaves no scale for
        # a relative drop -- is not a flat neighbourhood, it is an unanswered
        # question, and has to read differently from "checked and flat".
        'unmeasured': sorted(k for k, v in dims.items() if not v.get('evaluated', True)),
        'warn': any(v.get('flags_spike') for v in dims.values()),
    }


# ---------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------

def validate_strategy(
    *,
    strategy_cls: type,
    symbols,
    params: dict = None,
    start: str = None,
    end: str = None,
    cash: float = 100_000.0,
    slippage: float = 0.0,
    n_folds: int = 4,
    embargo: int = 10,
    holdout_frac: float = 0.0,
    n_periods: int = 0,
    trials_tried: int = None,
    bootstrap_draws: int = 2000,
    bootstrap_block: int = 20,
    seed: int = 42,
    skip: tuple = (),
    results_dir: str = None,
    update_data: bool = False,
    market=None,
) -> dict:
    """Run the checks and write a JSON report.

    ``params``: what to validate. Merged over the strategy's own defaults, so
    passing ``None`` validates the defaults as they stand.

    ``market``: pass a pre-built ``MarketData`` (e.g. in tests, with synthetic
    data) to skip loading from ``DataManager`` entirely; ``start``/``end`` are
    then unused except as report metadata.

    ``trials_tried``: how many configurations were tried before settling on
    these. Feeds the Deflated Sharpe's multiple-testing correction, which
    otherwise assumes only the neighbourhood grid was ever looked at.

    ``skip``: names from ``CHECKS`` to leave out.
    """
    if market is None:
        if start is None or end is None:
            raise ValueError('Provide either `market`, or both `start` and `end` to load one.')
        market = load_market(symbols, start, end, update=update_data)

    unknown = sorted(set(skip) - set(CHECKS))
    if unknown:
        raise ValueError(f"Unknown check(s) to skip: {unknown}. Known: {list(CHECKS)}")
    skip = set(skip)

    defaults = dict(getattr(strategy_cls, 'params', {}) or {})
    params = {**defaults, **(params or {})}
    if not check_constraints(strategy_cls, params):
        raise ValueError(
            f"{strategy_cls.__name__} rejects {params} via its own `constraints` -- "
            f"there is nothing to validate until the parameters are self-consistent."
        )

    space = resolve_space(strategy_cls)
    grid = build_grid(strategy_cls, space, params)
    reserve = _reserve_bars(market, strategy_cls, grid)

    folds, holdout = anchored_walk_forward(
        market.n_bars, reserve_bars=reserve, n_folds=n_folds,
        embargo=embargo, holdout_frac=holdout_frac,
    )
    full = Window('full', reserve, holdout.start)

    logger.info(
        '%s: %d bars, reserve=%d, %d folds, embargo=%d, grid=%d candidate(s)',
        strategy_cls.__name__, market.n_bars, reserve, n_folds, embargo, len(grid),
    )

    def run(window: Window, candidate: dict = None):
        return run_window(
            market, strategy_cls, candidate or params, window,
            cash=cash, slippage=slippage, pad=reserve,
        )

    base_out = run(full)
    base_metrics = base_out['metrics']
    if not base_metrics:
        raise RuntimeError(
            f'{strategy_cls.__name__} recorded no bars over [{full.start}, {full.end}) -- '
            f'there is nothing to validate. Most often the indicator warmup exceeds '
            f'the loaded date range.'
        )

    base_returns = _returns_over(full, base_out)

    # One run per candidate, read by both the sensitivity check and the
    # PBO/DSR matrix. `plateau_check` asks for the base set first and then
    # each neighbour, so without memoizing, the base set alone would be
    # backtested once per dimension.
    grid_cache: dict = {
        _params_key(params): (
            score(base_metrics, window_years=window_years(full.n_bars)), base_returns,
        ),
    }

    def evaluate(candidate: dict) -> float:
        key = _params_key(candidate)
        if key not in grid_cache:
            out = run(full, candidate)
            grid_cache[key] = (
                score(out['metrics'], window_years=window_years(full.n_bars)),
                _returns_over(full, out),
            )
        return grid_cache[key][0]

    checks: dict = {}

    if 'walkforward' not in skip:
        checks['walkforward'] = _check_walkforward(run, folds)

    if 'subperiod' not in skip:
        checks['subperiod'] = _check_subperiod(run, market, reserve, full.end, n_periods)

    needs_grid = {'sensitivity', 'pbo', 'dsr'} - skip
    if needs_grid:
        # Fill the cache for every candidate even when only PBO/DSR were
        # asked for: the matrix has to be complete or the two halves of the
        # neighbourhood disagree about what was compared.
        for candidate in grid:
            evaluate(candidate)

    if 'sensitivity' not in skip:
        checks['sensitivity'] = _check_sensitivity(strategy_cls, space, params, evaluate)

    if 'bootstrap' not in skip:
        dates = [r['date'] for r in base_out['result']['equity_records']]
        boot = block_bootstrap(
            base_returns,
            n_draws=bootstrap_draws, block=bootstrap_block,
            periods_per_year=annualization_factor(dates), seed=seed,
        )
        boot['warn'] = not boot['insufficient'] and boot['sharpe']['p_positive'] < 0.95
        checks['bootstrap'] = boot

    if needs_grid & {'pbo', 'dsr'}:
        matrix = np.stack([grid_cache[_params_key(c)][1] for c in grid], axis=0)
        if 'pbo' not in skip:
            pbo = pbo_cscv(matrix, n_blocks=16)
            pbo['warn'] = bool(pbo['pbo'] > 0.5)   # NaN compares False: an unanswered check flags nothing
            pbo['note'] = (
                'Candidates are a one-step neighbourhood, not an independent '
                'search, so this reads as "would picking among these nearby '
                'variants in-sample have held up out of sample" -- narrower '
                'than PBO over a real search.'
            )
            checks['pbo'] = pbo
        if 'dsr' not in skip:
            dsr = deflated_sharpe_ratio(matrix, 0, n_trials=trials_tried or len(grid))
            dsr['warn'] = bool(dsr['dsr'] < 0.95)
            dsr['trials_declared'] = trials_tried is not None
            checks['dsr'] = dsr

    flags = sorted(name for name, result in checks.items() if result.get('warn'))

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'strategy': strategy_cls.__name__,
        'strategy_key': name_for(strategy_cls),
        'timestamp': ts,
        'symbols': sorted(symbols),
        'start': start,
        'end': end,
        'cash': cash,
        'slippage': slippage,
        'n_bars': market.n_bars,
        'reserve_bars': reserve,
        'n_folds': n_folds,
        'embargo': embargo,
        'holdout_frac': holdout_frac,
        'evaluation_window': {'start': full.start, 'end': full.end},
        # The key `ft.py backtest --params-from` reads, so a validated set can
        # be replayed with the full trade log and charts.
        'params': params,
        'space': {k: spec_to_json(v) for k, v in space.items()},
        'grid_size': len(grid),
        'trials_tried': trials_tried,
        'skipped': sorted(skip),
        'full_metrics': base_metrics,
        'checks': checks,
        'flags': flags,
    }

    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{strategy_cls.__name__}_{ts}_validation.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info('Report written: %s', path)
    return {'path': path, 'report': report}
