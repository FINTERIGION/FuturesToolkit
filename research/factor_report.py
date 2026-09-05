"""Orchestration: turn ``research.factor_eval`` primitives into one
JSON-shaped report, identical whether produced by ``factor_runner.py`` or
the web panel.

Kept separate from ``factor_eval`` on purpose -- that module's own rule is
pure stat functions with no I/O and no concrete-``Factor`` awareness (see
its docstring). This one assembles those primitives plus an actual
``Factor`` instance into the single report dict ``factor_runner.py``'s
``report``/``corr`` subcommands and ``web/routers/factors.py``'s
``/api/factors/report``/``/api/factors/corr`` endpoints both write and
read -- one shape, so a report saved by the CLI renders the same charts a
web-panel run would, and vice versa (the same reasoning behind
``core.backtest.run_single_backtest`` living in ``core`` rather than behind
one particular CLI). ``save_report`` is the one place either side writes a
report to disk, under the same ``results/factors/`` directory and naming
convention, so a fresh CLI run and a fresh web-panel run always list
together in ``GET /api/factors/reports``.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from typing import Dict, Mapping, Sequence

import numpy as np

from factors.base import Factor, FactorContext, FactorPanel, compute_factor

from .factor_corr import correlation_matrix, ic_correlation, to_rows
from .factor_eval import (
    annual_slices, coverage_summary, cumulative_curves, factor_autocorr,
    factor_turnover, forward_returns, ic_decay, ic_series, quantile_returns,
)

__all__ = ['build_factor_panel', 'build_report', 'build_corr_report', 'save_report', 'RESULTS_DIR']

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'factors')


def build_factor_panel(factor: Factor, market, symbols: Sequence[str] = None) -> FactorPanel:
    """Compute ``factor`` over ``market``, defaulting to its full symbol list."""
    ctx = FactorContext(market, symbols if symbols is not None else market.symbols)
    return compute_factor(factor, ctx)


def _dates_iso(dates) -> list:
    return [str(d) for d in dates]


def build_report(
    market,
    factor: Factor,
    *,
    horizons: Sequence[int],
    n_groups: int = 3,
    return_source: str = 'weighted',
) -> dict:
    """Full single-factor evaluation: coverage, IC decay, annual stability,
    quantile buckets, turnover, autocorrelation -- plus the raw curves a
    chart needs (cumulative IC, quantile-bucket net value) that a
    print-only report has no reason to keep.
    """
    panel = build_factor_panel(factor, market)
    fwds = forward_returns(market, horizons, lag=1, source=return_source, symbols=panel.symbols)
    decay = ic_decay(panel, fwds)

    primary_h = int(horizons[0])
    primary_fwd = fwds[primary_h]
    primary_ic = ic_series(panel, primary_fwd)
    quant = quantile_returns(panel, primary_fwd, n_groups, primary_h)
    turnover = factor_turnover(panel, n_groups, rebalance=primary_h)
    autocorr = factor_autocorr(panel)
    coverage = coverage_summary(panel)
    slices = annual_slices(panel, primary_fwd, market.dates, horizon=primary_h, n_groups=n_groups)

    cum_ic = np.nancumsum(np.nan_to_num(primary_ic, nan=0.0))
    rows, curves = cumulative_curves(quant['group_returns'], primary_h)

    return {
        'factor': type(factor).__name__,
        'params': dict(factor.p),
        'symbols': panel.symbols,
        'return_source': return_source,
        'n_groups': n_groups,
        'horizons': [int(h) for h in horizons],
        'coverage': coverage,
        'ic_decay': {str(h): s for h, s in decay.items()},
        'annual_slices': {str(y): r for y, r in slices.items()},
        'quantiles': {
            'horizon': primary_h,
            'groups': {str(g): s for g, s in quant['groups'].items()},
            'spread': quant['spread'],
            'monotonicity': quant['monotonicity'],
        },
        'turnover': turnover,
        'autocorr': {str(k): v for k, v in autocorr.items()},
        'ic_curve': {'dates': _dates_iso(market.dates), 'cumulative_ic': cum_ic.tolist()},
        'quantile_curve': {
            'dates': _dates_iso(market.dates[rows]),
            'curves': curves.tolist(),
        },
    }


def build_corr_report(market, factor_panels: Mapping[str, FactorPanel], *, horizon: int) -> dict:
    """Cross-sectional and IC correlation between every factor in ``factor_panels``.

    ``factor_panels`` must all be computed over the same ``market.symbols``
    -- the same shape requirement :func:`research.factor_corr.correlation_matrix`
    itself enforces.
    """
    cs_result = correlation_matrix(factor_panels, method='spearman')
    fwd = forward_returns(market, [horizon], symbols=market.symbols)[horizon]
    ics = {name: ic_series(panel, fwd) for name, panel in factor_panels.items()}
    ic_result = ic_correlation(ics)
    return {
        'symbols': list(market.symbols),
        'horizon': int(horizon),
        'correlation_matrix': to_rows(cs_result),
        'ic_correlation': to_rows(ic_result),
    }


def _sanitize(obj):
    """Recursively make ``obj`` valid, finite JSON.

    A plain Python ``float('nan')``/``inf`` serializes just fine as far as
    ``json.dump`` is concerned -- it emits the bare (non-standard) tokens
    ``NaN``/``Infinity`` rather than raising, so ``default=`` is never even
    called for it. Both that case and any numpy scalar/array that slipped
    through ``build_report``/``build_corr_report`` (a caller-supplied
    ``params`` dict can carry either) are walked and normalized eagerly here
    instead, the same problem ``web.serialize.jsonable`` solves for the API
    layer -- duplicated rather than imported, since ``research`` has no
    business depending on ``web``.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def save_report(payload: dict, name_prefix: str, results_dir: str = None) -> str:
    """Write ``payload`` as timestamped JSON under ``results_dir`` (default
    :data:`RESULTS_DIR`), naming it ``{name_prefix}_{YYYYmmdd_HHMMSS}.json`` --
    the same ``_{timestamp}.`` convention ``runner.py --keep-last`` and every
    other results-directory writer in this codebase already relies on.
    """
    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'{name_prefix}_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(payload), f, indent=2)
    return path
