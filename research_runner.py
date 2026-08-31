"""
FuturesToolkit research CLI: Optuna parameter optimization and meta-label
signal filtering, both strategy-agnostic -- every subcommand works on any
strategy ``strategies.discover_strategies()`` finds, with no code changes.

Usage (from the repo root):
  python research_runner.py show-space --strategy double_ma
  python research_runner.py optimize --strategy double_ma --n-trials 200
  python research_runner.py holdout --best results/optuna/DoubleMaStrategy_..._best.json
  python research_runner.py metalabel --strategy double_ma --params-from results/optuna/..._best.json
  python research_runner.py meta-backtest --model results/meta/DoubleMaStrategy_....joblib

Run ``python research_runner.py <subcommand> --help`` for each command's full flag list.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

import joblib

from datafeed.products import list_products
from runner import parse_param_value
from strategies import discover_strategies, load_strategy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_META_DIR = os.path.join(ROOT_DIR, 'results', 'meta')

logger = logging.getLogger('futurestoolkit.research')


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(message)s')


def _strategy_choices() -> list:
    return sorted(discover_strategies())


def _load_params_from_report(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)['best_params']


def _resolve_params(strategy_cls: type, args) -> dict:
    params = dict(getattr(strategy_cls, 'params', {}) or {})
    if getattr(args, 'params_from', None):
        params.update(_load_params_from_report(args.params_from))
    for raw in getattr(args, 'param', None) or []:
        name, value = parse_param_value(raw)
        params[name] = value
    return params


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
# metalabel
# ---------------------------------------------------------------------

def cmd_metalabel(args) -> None:
    from research.metalabel import fit_meta_model
    from research.runner_api import load_market

    strategy_cls = load_strategy(args.strategy)
    params = _resolve_params(strategy_cls, args)
    logger.info('Strategy params: %s', params)

    market = load_market(args.symbols, args.start, args.end, update=args.update_data)
    bundle = fit_meta_model(
        market, strategy_cls, params,
        cash=args.cash, slippage=args.slippage,
        n_folds=args.n_folds, embargo=args.embargo, holdout_frac=args.holdout_frac,
        kind=args.model, max_features=args.max_features, seed=args.seed,
        min_events_per_fold=args.min_events_per_fold,
    )
    bundle['symbols'] = sorted(args.symbols)
    bundle['start'] = args.start
    bundle['end'] = args.end

    logger.info(
        'Fitted on %d/%d usable events (%d out-of-fold) -- OOF AUC=%.3f, threshold=%.2f, '
        'expectancy@threshold=%.2f (n_kept=%d, keep_frac=%.0f%%)',
        bundle['n_events_usable'], bundle['n_events_total'], bundle['n_events_oof'],
        bundle['oof_auc'], bundle['threshold'], bundle['threshold_info']['expectancy'],
        bundle['threshold_info']['n_kept'], bundle['threshold_info']['keep_frac'] * 100,
    )

    out_dir = args.out_dir or DEFAULT_META_DIR
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'{strategy_cls.__name__}_{ts}.joblib')
    joblib.dump(bundle, path)
    logger.info('Bundle written: %s', path)
    print(f'Next: python research_runner.py meta-backtest --model {path}')


# ---------------------------------------------------------------------
# meta-backtest
# ---------------------------------------------------------------------

def cmd_meta_backtest(args) -> None:
    from research.metalabel import evaluate_meta_backtest
    from research.runner_api import load_market

    bundle = joblib.load(args.model)
    market = load_market(bundle['symbols'], bundle['start'], bundle['end'])
    res = evaluate_meta_backtest(bundle, market, n_random=args.n_random, seed=args.seed)

    base, gated, rb = res['baseline_metrics'], res['gated_metrics'], res['random_baseline']
    print(f"Evaluation window: bars [{res['window']['start']}, {res['window']['end']})")
    print(f"{'':16}{'baseline':>14}{'gated':>14}")
    for key, label in [
        ('sharpe_ratio', 'Sharpe'), ('max_drawdown', 'Max DD %'),
        ('n_trades', 'N trades'), ('win_rate', 'Win rate %'), ('expectancy', 'Expectancy'),
    ]:
        print(f'{label:16}{base.get(key, 0):>14.4f}{gated.get(key, 0):>14.4f}')
    print()
    print(f"OOF coverage: the filter scored {res['n_events_scored']} of the "
          f"{res['n_events_in_window']} entry signals in this window "
          f"({res['oof_coverage'] * 100:.0f}%); the rest carried no out-of-fold "
          f"prediction and were passed through untouched in both arms.")
    if res['oof_coverage'] < 0.5:
        print('  *** Fewer than half the signals were scored -- this verdict rests on few '
              'decisions. Widen the folds or lower min_train_events in oof_predict. ***')

    gc = res['gate_counters']
    print(f"Rejected {res['n_rejected']}/{res['n_events_scored']} scored entries "
          f"(kept {res['n_kept']}); the gate flattened {gc['blocked_total']} entry "
          f"attempt(s) over the run.")
    if gc['blocked_total'] != gc['threshold_rejected']:
        # Should be unreachable while both arms run with on_missing='pass' --
        # if it ever fires, gated-vs-baseline has stopped being a measurement
        # of the filter alone, which is exactly the failure worth shouting about.
        print(f"  *** {gc['blocked_total'] - gc['threshold_rejected']} of those were "
              f"blocked for a reason other than the P(win) threshold -- gated vs. "
              f"baseline is NOT a clean measurement of the filter. ***")

    if res['n_rejected'] == 0:
        print('Sum of net_pnl on rejected trades: n/a -- the filter rejected nothing.')
    else:
        print(f"Sum of net_pnl on rejected trades: {res['rejected_net_pnl']:,.2f} "
              f"{'*** POSITIVE -- the filter is cutting winners ***' if res['rejected_net_pnl_positive'] else '(negative -- filter is cutting losers, as intended)'}")

    if rb['degenerate']:
        print('Random-rejection baseline: not applicable -- with 0 rejections every draw '
              'reproduces the gated run itself, so there is nothing to compare against.')
    else:
        print(f"Random-rejection baseline ({rb['n_random']} draws at the same rejection rate): "
              f"mean Sharpe={rb['sharpe_mean']:.4f}, std={rb['sharpe_std']:.4f}")
        print(f"Gated Sharpe={rb['gated_sharpe']:.4f} sits at the {rb['gated_percentile']:.0f}th percentile "
              f"of random rejection -- {'BEATS random (>=90th pct)' if rb['beats_random'] else 'does NOT clearly beat random'}")
    print(f"Out-of-fold AUC: {res['oof_auc']:.4f}")


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

def _add_data_args(p, default_symbols=('SA', 'FG', 'CF', 'MA', 'TA', 'SR', 'OI')):
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

    p = sub.add_parser('metalabel', help='Fit a meta-label filter (purged walk-forward) for a strategy.')
    p.add_argument('--strategy', required=True, choices=_strategy_choices())
    _add_data_args(p)
    _add_split_args(p)
    p.add_argument('--model', choices=['logit', 'rf'], default='logit')
    p.add_argument('--max-features', type=int, default=12)
    p.add_argument('--min-events-per-fold', type=int, default=20)
    p.add_argument('--params-from', default=None, help='Load strategy params from an optimize *_best.json report.')
    p.add_argument('--param', action='append', help='Concrete strategy param override: name=value')
    p.add_argument('--out-dir', default=None)
    p.set_defaults(func=cmd_metalabel)

    p = sub.add_parser('meta-backtest', help='Evaluate a fitted meta-label bundle: gated vs. baseline vs. random.')
    p.add_argument('--model', required=True, help='Path to a metalabel .joblib bundle.')
    p.add_argument('--n-random', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_meta_backtest)

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
