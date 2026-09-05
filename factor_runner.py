"""
FuturesToolkit factor CLI: judge a predictive score directly -- IC, quantile
buckets, decay, turnover, correlation -- without running a backtest, and
without spending sizing, margin, commission, or the strategy-tuning holdout
window to find out whether a score carries any information at all.

Usage (from the repo root):
  python factor_runner.py list
  python factor_runner.py show-space --factor momentum
  python factor_runner.py ic --factor momentum --horizons 1 5 20 --return-source exec
  python factor_runner.py quantiles --factor carry --n-groups 3
  python factor_runner.py report --factor momentum --n-groups 3
  python factor_runner.py corr

Once a factor looks worth trading, ``strategies.factor_bridge`` turns it into
a runnable, tunable strategy for free::

  python runner.py --strategy factor_carry
  python research_runner.py optimize --strategy factor_carry

Run ``python factor_runner.py <subcommand> --help`` for each command's full
flag list.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from datafeed.products import list_products

logger = logging.getLogger('futurestoolkit.factors')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'factors')

DEFAULT_SYMBOLS = ['SA', 'FG', 'CF', 'C']
DEFAULT_START = '2016-01-01'
DEFAULT_END = '2026-12-31'
DEFAULT_HORIZONS = [1, 5, 10, 20]
DEFAULT_N_GROUPS = 3
DEFAULT_HORIZON = 5


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(message)s')


def _factor_choices() -> list:
    from factors import discover_factors
    return sorted(discover_factors())


def _parse_param_value(raw: str) -> tuple:
    """``name=value`` -> ``(name, value)``, casting to int, then float, then
    bool, else leaving it a string. Same grammar as ``runner.py --param``
    (a point value), not ``optimize``'s ``name=kind:args`` search range."""
    if '=' not in raw:
        raise ValueError(f"Invalid --param {raw!r}; expected name=value")
    name, value = raw.split('=', 1)
    for caster in (int, float):
        try:
            return name, caster(value)
        except ValueError:
            continue
    if value.lower() in ('true', 'false'):
        return name, value.lower() == 'true'
    return name, value


def _build_factor(args):
    from factors import load_factor

    factor_cls = load_factor(args.factor)
    overrides = dict(_parse_param_value(p) for p in (getattr(args, 'param', None) or []))
    unknown = set(overrides) - set(factor_cls.params or {})
    if unknown:
        logger.warning(
            "%s does not declare param(s) %s -- passed through anyway (typo?)",
            factor_cls.__name__, sorted(unknown),
        )
    return factor_cls(**overrides)


def _load_market(args):
    from research.runner_api import load_market
    return load_market(args.symbols, args.start, args.end, update=args.update_data)


def _compute_panel(args):
    from factors.base import FactorContext, compute_factor

    market = _load_market(args)
    factor = _build_factor(args)
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(factor, ctx)
    return market, factor, panel


def _write_report(name: str, payload: dict, results_dir: str) -> str:
    from research.factor_report import save_report
    path = save_report(payload, name, results_dir)
    logger.info('Wrote %s', path)
    return path


# ---------------------------------------------------------------------
# printing helpers
# ---------------------------------------------------------------------

def _print_coverage(coverage: dict) -> None:
    print(
        f'coverage: {coverage["mean"]:.1f}/{coverage["n_symbols"]} avg symbols scored, '
        f'first scored bar {coverage["first_scored_bar"]}, '
        f'{coverage["bars_with_full_coverage"]} bar(s) with full coverage'
    )


def _print_ic_table(decay: dict) -> None:
    print(f'{"horizon":>8} {"n_obs":>8} {"mean":>9} {"ir":>8} {"t_stat":>8} {"p_value":>9} {"pos_rate":>9}')
    for h, s in decay.items():
        print(
            f'{h:>8} {s["n_obs"]:>8} {s["mean"]:>9.4f} {s["ir"]:>8.3f} '
            f'{s["t_stat"]:>8.2f} {s["p_value"]:>9.4f} {s["positive_rate"]:>9.2%}'
        )


def _print_annual_table(slices: dict) -> None:
    print('annual slices -- never trust the aggregate row above without this:')
    print(f'{"year":>6} {"n_bars":>7} {"ic_mean":>9} {"ic_ir":>8} {"ls_sharpe":>10}')
    for year, row in slices.items():
        print(
            f'{year:>6} {row["n_bars"]:>7} {row["ic_mean"]:>9.4f} '
            f'{row["ic_ir"]:>8.3f} {row["long_short_sharpe"]:>10.3f}'
        )


def _print_quantile_table(result: dict) -> None:
    groups = result['groups']
    print(f'{"group":>6} {"n_obs":>8} {"mean":>9} {"sharpe":>8}')
    for g in sorted(groups, key=int):
        s = groups[g]
        print(f'{g:>6} {s["n_obs"]:>8} {s["mean"]:>9.5f} {s["sharpe"]:>8.3f}')
    spread = result['spread']
    print(f'{"spread":>6} {spread["n_obs"]:>8} {spread["mean"]:>9.5f} {spread["sharpe"]:>8.3f}')
    print(f'monotonicity: {result["monotonicity"]:.3f}')


def _print_turnover_autocorr(turnover: dict, autocorr: dict) -> None:
    print(f"turnover (top/bottom bucket, per rebalance): top={turnover['top']:.2%} bottom={turnover['bottom']:.2%}")
    ac = ', '.join(f'{lag}={v:.3f}' for lag, v in autocorr.items())
    print(f'factor autocorrelation -- {ac}')


def _rows_to_matrix(rows: list):
    """Reconstruct ``(names, matrix)`` from a ``research.factor_corr.to_rows()``
    list -- the JSON-friendly shape ``build_corr_report`` returns, and the
    one shape both the terminal table and the heatmap plot need."""
    import numpy as np

    names = [r['factor'] for r in rows]
    matrix = np.array(
        [[np.nan if r[n] is None else r[n] for n in names] for r in rows], dtype='float64',
    )
    return names, matrix


def _print_matrix(rows: list) -> None:
    names, matrix = _rows_to_matrix(rows)
    label_w = max(14, max((len(n) for n in names), default=0) + 2)
    col_w = max(10, max((len(n) for n in names), default=0) + 2)
    print(' ' * label_w + ''.join(f'{n:>{col_w}}' for n in names))
    for i, name in enumerate(names):
        cells = ''.join(
            f'{"nan":>{col_w}}' if matrix[i, j] != matrix[i, j] else f'{matrix[i, j]:>{col_w}.3f}'
            for j in range(len(names))
        )
        print(f'{name:<{label_w}}{cells}')


# ---------------------------------------------------------------------
# plotting helpers (matplotlib, Agg backend -- see plotting.py)
# ---------------------------------------------------------------------

def _plot_ic(name, dates, cum_ic, report_path) -> None:
    """``cum_ic`` is already cumulative -- callers compute it once (from a raw
    IC series, or read straight off a saved report's ``ic_curve``) so this
    stays a plain curve plot, reusable from either source."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from plotting import COLOR_EQ  # noqa: F401 (imports the dark theme rcParams too)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pd.to_datetime(dates), cum_ic, color=COLOR_EQ, linewidth=1.5)
    ax.set_title(f'{name}: cumulative IC')
    ax.grid(True, alpha=0.3)
    out_path = report_path.rsplit('.', 1)[0] + '_ic.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def _plot_quantiles(name, dates, curves, report_path) -> None:
    """``dates`` is already the per-row (subsampled, one per independent
    rebalance) date array matching ``curves``' rows -- see
    ``research.factor_eval.cumulative_curves``."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from plotting import COLOR_UP  # noqa: F401 (imports the dark theme rcParams too)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = pd.to_datetime(dates)
    for g in range(curves.shape[1]):
        ax.plot(x, curves[:, g], label=f'group {g}', linewidth=1.3)
    ax.legend()
    ax.set_title(f'{name}: quantile bucket cumulative return')
    ax.grid(True, alpha=0.3)
    out_path = report_path.rsplit('.', 1)[0] + '_quantiles.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def _plot_heatmap(rows, title, report_path, suffix) -> None:
    import matplotlib.pyplot as plt
    from plotting import COLOR_EQ  # noqa: F401 (imports the dark theme rcParams too)

    names, matrix = _rows_to_matrix(rows)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 2))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdYlGn')
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if v == v:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    out_path = report_path.rsplit('.', 1)[0] + f'{suffix}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


# ---------------------------------------------------------------------
# list
# ---------------------------------------------------------------------

def cmd_list(args) -> None:
    from factors import discover_factors

    registry = discover_factors()
    if not registry:
        print('No factors found under factors/.')
        return
    for name in sorted(registry):
        cls = registry[name]
        first_line = (cls.__doc__ or '').strip().splitlines()[:1]
        first_line = first_line[0] if first_line else ''
        params = ', '.join(f'{k}={v}' for k, v in (cls.params or {}).items())
        print(f'{name:<16} direction={cls.direction:+d}  {first_line}')
        if params:
            print(f'{"":<16} params: {params}')


# ---------------------------------------------------------------------
# show-space
# ---------------------------------------------------------------------

def cmd_show_space(args) -> None:
    from research.space import resolve_space
    from factors import load_factor

    factor_cls = load_factor(args.factor)
    space = resolve_space(factor_cls)
    print(f'{factor_cls.__name__} search space:')
    for name, spec in space.items():
        print(f'  {name}: {spec}')
    fixed = set(getattr(factor_cls, 'fixed_params', ()) or ())
    untuned = [k for k in (factor_cls.params or {}) if k not in space and k not in fixed]
    if fixed:
        print(f'  (fixed, never tuned: {sorted(fixed)})')
    if untuned:
        print(f'  (no space could be inferred, left at default: {untuned})')


# ---------------------------------------------------------------------
# ic
# ---------------------------------------------------------------------

def cmd_ic(args) -> None:
    from research.factor_eval import forward_returns, ic_series, ic_decay, coverage_summary, annual_slices

    market, factor, panel = _compute_panel(args)
    fwds = forward_returns(market, args.horizons, lag=1, source=args.return_source, symbols=panel.symbols)
    decay = ic_decay(panel, fwds)
    primary_h = args.horizons[0]
    primary_ic = ic_series(panel, fwds[primary_h])
    coverage = coverage_summary(panel)
    slices = annual_slices(panel, fwds[primary_h], market.dates, horizon=primary_h, n_groups=args.n_groups)

    print(f'{factor.__class__.__name__} -- IC (source={args.return_source}, params={factor.p})')
    _print_coverage(coverage)
    _print_ic_table(decay)
    print()
    _print_annual_table(slices)

    payload = {
        'factor': factor.__class__.__name__, 'params': factor.p,
        'symbols': panel.symbols, 'start': args.start, 'end': args.end,
        'return_source': args.return_source, 'horizons': list(args.horizons),
        'coverage': coverage,
        'ic_decay': {str(h): s for h, s in decay.items()},
        'annual_slices': {str(y): r for y, r in slices.items()},
    }
    path = _write_report(f'{args.factor}_ic', payload, args.results_dir)

    if not args.no_plots:
        import numpy as np
        cum_ic = np.nancumsum(np.nan_to_num(primary_ic, nan=0.0))
        _plot_ic(factor.__class__.__name__, market.dates, cum_ic, path)


# ---------------------------------------------------------------------
# quantiles
# ---------------------------------------------------------------------

def cmd_quantiles(args) -> None:
    from research.factor_eval import forward_returns, quantile_returns, cumulative_curves

    market, factor, panel = _compute_panel(args)
    h = args.horizons[0]
    fwd = forward_returns(market, [h], lag=1, source=args.return_source, symbols=panel.symbols)[h]
    result = quantile_returns(panel, fwd, args.n_groups, h)

    print(f'{factor.__class__.__name__} -- quantile returns (n_groups={args.n_groups}, horizon={h})')
    _print_quantile_table(result)

    payload = {
        'factor': factor.__class__.__name__, 'params': factor.p,
        'symbols': panel.symbols, 'start': args.start, 'end': args.end,
        'return_source': args.return_source, 'n_groups': args.n_groups, 'horizon': h,
        'groups': {str(g): s for g, s in result['groups'].items()},
        'spread': result['spread'], 'monotonicity': result['monotonicity'],
    }
    path = _write_report(f'{args.factor}_quantiles', payload, args.results_dir)

    if not args.no_plots:
        rows, curves = cumulative_curves(result['group_returns'], h)
        _plot_quantiles(factor.__class__.__name__, market.dates[rows], curves, path)


# ---------------------------------------------------------------------
# report (ic + quantiles + turnover + autocorr, one JSON)
# ---------------------------------------------------------------------

def cmd_report(args) -> None:
    from research.factor_report import build_report

    market = _load_market(args)
    factor = _build_factor(args)
    report = build_report(
        market, factor, horizons=args.horizons, n_groups=args.n_groups,
        return_source=args.return_source,
    )

    print(f'{report["factor"]} -- full report (source={report["return_source"]}, params={report["params"]})')
    _print_coverage(report['coverage'])
    _print_ic_table(report['ic_decay'])
    print()
    _print_annual_table(report['annual_slices'])
    print()
    print(f'quantile returns (n_groups={report["n_groups"]}, horizon={report["quantiles"]["horizon"]}):')
    _print_quantile_table(report['quantiles'])
    print()
    _print_turnover_autocorr(report['turnover'], report['autocorr'])

    payload = {**report, 'start': args.start, 'end': args.end}
    path = _write_report(f'{args.factor}_report', payload, args.results_dir)

    if not args.no_plots:
        import numpy as np
        _plot_ic(report['factor'], market.dates, np.array(report['ic_curve']['cumulative_ic']), path)
        qc = report['quantile_curve']
        _plot_quantiles(report['factor'], qc['dates'], np.array(qc['curves']), path)


# ---------------------------------------------------------------------
# corr
# ---------------------------------------------------------------------

def cmd_corr(args) -> None:
    from factors import discover_factors
    from research.factor_report import build_factor_panel, build_corr_report

    registry = discover_factors()
    if len(registry) < 2:
        raise ValueError(f'corr needs at least 2 discovered factors under factors/; found {sorted(registry)}')

    market = _load_market(args)
    panels = {name: build_factor_panel(factor_cls(), market) for name, factor_cls in sorted(registry.items())}
    result = build_corr_report(market, panels, horizon=args.horizon)

    print('Cross-sectional rank correlation -- do factors rank products the same way?')
    _print_matrix(result['correlation_matrix'])
    print('\nIC correlation -- do factors work at the same times?')
    _print_matrix(result['ic_correlation'])

    payload = {**result, 'start': args.start, 'end': args.end}
    path = _write_report('corr', payload, args.results_dir)

    if not args.no_plots:
        _plot_heatmap(result['correlation_matrix'], 'Cross-sectional correlation', path, '_cs')
        _plot_heatmap(result['ic_correlation'], 'IC correlation', path, '_ic_corr')


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

def _add_data_args(p, default_symbols=DEFAULT_SYMBOLS) -> None:
    p.add_argument('--symbols', nargs='+', default=list(default_symbols),
                    help=f'Products to load (registered: {", ".join(list_products())})')
    p.add_argument('--start', default=DEFAULT_START)
    p.add_argument('--end', default=DEFAULT_END)
    p.add_argument('--update-data', action='store_true')


def _add_eval_args(p) -> None:
    p.add_argument('--horizons', nargs='+', type=int, default=list(DEFAULT_HORIZONS),
                    help='Forward-return horizons in bars, e.g. --horizons 1 5 20')
    p.add_argument('--return-source', choices=('weighted', 'exec'), default='weighted',
                    help="'weighted' (default): OI-weighted continuous series, gap-free. "
                         "'exec': the real calendar contract, NaN across a roll.")
    p.add_argument('--n-groups', type=int, default=DEFAULT_N_GROUPS,
                    help='Quantile buckets for the cross-section (default: %(default)s; '
                         'keep this small on a short symbol list).')
    p.add_argument('--param', action='append',
                    help="Factor param override: name=value, e.g. --param lookback=90")
    p.add_argument('--no-plots', action='store_true', help='Skip writing PNG charts.')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FuturesToolkit factor CLI.')
    parser.add_argument('--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('list', help='List every discovered factor.')
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('show-space', help="Print a factor's tunable search space.")
    p.add_argument('--factor', required=True, choices=_factor_choices())
    p.set_defaults(func=cmd_show_space)

    p = sub.add_parser('ic', help='Information coefficient: level, decay, annual stability.')
    p.add_argument('--factor', required=True, choices=_factor_choices())
    _add_data_args(p)
    _add_eval_args(p)
    p.set_defaults(func=cmd_ic)

    p = sub.add_parser('quantiles', help='Sort the cross-section into buckets and measure what each earned.')
    p.add_argument('--factor', required=True, choices=_factor_choices())
    _add_data_args(p)
    _add_eval_args(p)
    p.set_defaults(func=cmd_quantiles)

    p = sub.add_parser('report', help='Full single-factor evaluation: ic + quantiles + turnover + autocorrelation.')
    p.add_argument('--factor', required=True, choices=_factor_choices())
    _add_data_args(p)
    _add_eval_args(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('corr', help='Cross-sectional and IC correlation between every discovered factor.')
    _add_data_args(p)
    p.add_argument('--horizon', type=int, default=DEFAULT_HORIZON)
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    p.set_defaults(func=cmd_corr)

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    try:
        args.func(args)
    except (ValueError, RuntimeError) as e:
        logger.error('%s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
