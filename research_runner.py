"""
FuturesToolkit research CLI: Optuna parameter optimization, strategy-agnostic
-- every subcommand works on any strategy ``strategies.discover_strategies()``
finds, with no code changes.

Usage (from the repo root):
  python research_runner.py show-space --strategy double_ma
  python research_runner.py optimize --strategy double_ma --n-trials 200
  python research_runner.py holdout --best results/optuna/DoubleMaStrategy_..._best.json

Run ``python research_runner.py <subcommand> --help`` for each command's full flag list.
"""

from __future__ import annotations

import argparse
import logging
import sys

from datafeed.products import list_products
from strategies import discover_strategies, load_strategy

logger = logging.getLogger('futurestoolkit.research')


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(message)s')


def _strategy_choices() -> list:
    return sorted(discover_strategies())


# ---------------------------------------------------------------------
# show-space
# ---------------------------------------------------------------------

def cmd_show_space(args) -> None:
    from research.space import resolve_space

    strategy_cls = load_strategy(args.strategy)
    space = resolve_space(strategy_cls)
    print(f'{strategy_cls.__name__} search space:')
    for name, spec in space.items():
        print(f'  {name}: {spec}')
    fixed = set(getattr(strategy_cls, 'fixed_params', ()) or ())
    untuned = [k for k in (strategy_cls.params or {}) if k not in space and k not in fixed]
    if fixed:
        print(f'  (fixed, never tuned: {sorted(fixed)})')
    if untuned:
        print(f'  (no space could be inferred, left at default: {untuned})')


# ---------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------

def cmd_optimize(args) -> None:
    from research.optimize import run_study
    from research.space import parse_param_override

    strategy_cls = load_strategy(args.strategy)
    overrides = dict(parse_param_override(p) for p in (args.param or []))

    out = run_study(
        strategy_cls=strategy_cls,
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        cash=args.cash,
        slippage=args.slippage,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        embargo=args.embargo,
        holdout_frac=args.holdout_frac,
        lambda_std=args.lambda_std,
        min_trades_per_year=args.min_trades_per_year,
        dd_cap=args.dd_cap,
        param_overrides=overrides or None,
        seed=args.seed,
        probe_samples=args.probe_samples,
        study_name=args.study_name,
        results_dir=args.results_dir,
        update_data=args.update_data,
    )
    # `run_study` already logs the report path at INFO; only the follow-up
    # hint is this layer's to print.
    print(f"Next: python research_runner.py holdout --best {out['path']}")


# ---------------------------------------------------------------------
# holdout
# ---------------------------------------------------------------------

def cmd_holdout(args) -> None:
    from research.optimize import evaluate_holdout

    evaluate_holdout(args.best, force=args.force)


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

def _add_data_args(p, default_symbols=('SA', 'FG', 'CF', 'BU', 'RB', 'HC',
                                       'C', 'JM', 'V')):
    p.add_argument('--symbols', nargs='+', default=list(default_symbols),
                    help=f'Products to load (registered: {", ".join(list_products())})')
    p.add_argument('--start', default='2020-01-01')
    p.add_argument('--end', default='2026-12-31')
    p.add_argument('--cash', type=float, default=100_000.0)
    p.add_argument('--slippage', type=float, default=0.0)
    p.add_argument('--update-data', action='store_true')


def _add_split_args(p):
    p.add_argument('--n-folds', type=int, default=4)
    p.add_argument('--embargo', type=int, default=10)
    p.add_argument('--holdout-frac', type=float, default=0.20)
    p.add_argument('--seed', type=int, default=42)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FuturesToolkit research CLI.')
    parser.add_argument('--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('show-space', help="Print a strategy's tunable search space.")
    p.add_argument('--strategy', required=True, choices=_strategy_choices())
    p.set_defaults(func=cmd_show_space)

    p = sub.add_parser('optimize', help='Optuna anchored walk-forward parameter search.')
    p.add_argument('--strategy', required=True, choices=_strategy_choices())
    _add_data_args(p)
    _add_split_args(p)
    p.add_argument('--n-trials', type=int, default=200)
    p.add_argument('--probe-samples', type=int, default=20)
    p.add_argument('--lambda-std', type=float, default=0.5)
    p.add_argument('--min-trades-per-year', type=float, default=4.0)
    p.add_argument('--dd-cap', type=float, default=0.35)
    p.add_argument('--param', action='append',
                    help='Search-space override: name=kind:args, e.g. slow_period=int:20:200')
    p.add_argument('--study-name', default=None)
    p.add_argument('--results-dir', default=None)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser('holdout', help='Evaluate an optimize report on its locked holdout window (once).')
    p.add_argument('--best', required=True, help='Path to an optimize *_best.json report.')
    p.add_argument('--force', action='store_true', help='Re-evaluate even if already evaluated.')
    p.set_defaults(func=cmd_holdout)

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
