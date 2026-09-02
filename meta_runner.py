"""
FuturesToolkit meta-labeling CLI: filter a strategy's entries with a
second-stage classifier, and emit today's filtered signal.

Strategy-agnostic -- every subcommand works on any strategy
``strategies.discover_strategies()`` finds.

Usage (from the repo root):
  python meta_runner.py harvest     --strategy double_ma
  python meta_runner.py walkforward --strategy double_ma --shuffle-control
  python meta_runner.py holdout     --report results/meta/<...>_walkforward.json --keep-rate 0.65
  python meta_runner.py fit         --strategy double_ma --keep-rate 0.65 -o models/dma.joblib
  python meta_runner.py signal      --model models/dma.joblib

The intended order is exactly that: ``harvest`` says whether there are enough
trades to train on at all, ``walkforward`` says whether the filter has any
out-of-sample edge, ``fit`` freezes a model once it does, and ``signal``
reads that model for tomorrow's orders. Skipping to ``fit`` will produce an
artifact, but nothing that justifies trading it.

``signal`` is an alias for ``live_runner.py --model``: the live-signal code
lives in ``live/`` because it is not a meta-labeling idea, and the same
module serves an ungated strategy from ``strategies/``.

Run ``python meta_runner.py <subcommand> --help`` for each command's flags.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np

from core.market import slice_market
from datafeed.products import list_products
from research.runner_api import load_market, run_window
from research.splits import Window, anchored_walk_forward
from research.warmup import probe_warmup
from runner import resolve_params, run_single_backtest
from strategies import discover_strategies, load_strategy
from strategies.base import SetupContext

from live.report import emit
from live.signal import compute_signal, spec_from_model

from meta.dataset import purged_train_mask, samples_from_trades, window_mask
from meta.evaluate import (
    classifier_report,
    compare_metrics,
    format_table,
    summarize_folds,
    verdict,
)
from meta.features import build_feature_arrays
from meta.filter import make_meta_filtered, unfiltered
from meta.model import (
    FittedMetaModel,
    WalkForwardMetaModel,
    fit_estimator,
    save_model,
    threshold_for,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'meta')
LIVE_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'live')
DEFAULT_SYMBOLS = ['SA', 'FG', 'CF', 'BU', 'RB', 'HC', 'C', 'JM', 'V']
DEFAULT_KEEP_RATES = [1.0, 0.8, 0.65, 0.5]
MIN_SAMPLES = 300

logger = logging.getLogger('futurestoolkit.meta')


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(message)s')


def _strategy_choices() -> list:
    return sorted(discover_strategies())


def _feature_arrays_for(market, strategy_cls, params) -> dict:
    """``build_feature_arrays`` against ``market``'s own bar numbering.

    Built through a throwaway ``SetupContext`` so the offline sample builder
    and the in-run gate compute features from exactly the same code path on
    exactly the same series.
    """
    from core.engine import Engine

    engine = Engine(market, strategy_cls(**params))
    return build_feature_arrays(SetupContext(engine))


def _harvest(market, strategy_cls, params, *, cash, slippage) -> dict:
    """One unfiltered full-span run through the wrapper, plus its samples.

    The baseline goes through ``unfiltered()`` rather than running
    ``strategy_cls`` bare so its warmup matches the filtered arm exactly --
    see ``meta.filter``.
    """
    base_wrapped = unfiltered(strategy_cls)
    outcome = run_single_backtest(market, base_wrapped, params, cash, slippage)
    arrays = _feature_arrays_for(market, strategy_cls, params)
    samples = samples_from_trades(outcome['result']['trade_logs'], arrays)
    return {'outcome': outcome, 'arrays': arrays, 'samples': samples}


def _sample_size_note(n: int) -> str:
    if n >= 800:
        return f'{n} samples: enough for the shallow random forest.'
    if n >= MIN_SAMPLES:
        return (f'{n} samples: thin. Use --kind lr; a tree on this many rows will '
                'mostly memorise.')
    return (f'{n} samples: too few to train anything. Meta-labeling does not apply to '
            'this strategy at this configuration.')


# ---------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------

def cmd_harvest(args) -> None:
    strategy_cls = load_strategy(args.strategy)
    params = resolve_params(strategy_cls, args)
    market = load_market(args.symbols, args.start, args.end, update=args.update_data)

    logger.info('Harvesting %s trades over %d bars ...', strategy_cls.__name__, market.n_bars)
    h = _harvest(market, strategy_cls, params, cash=args.cash, slippage=args.slippage)
    samples, metrics = h['samples'], h['outcome']['metrics']

    sep = '=' * 62
    logger.info('\n'.join([
        sep, '  Meta-label sample harvest', sep,
        f'  Strategy         : {strategy_cls.__name__}',
        f'  Window           : {args.start} -> {args.end}  ({market.n_bars} bars)',
        f'  Trades in log    : {metrics.get("n_trades", 0)}',
        f'  Usable samples   : {len(samples)}',
        f'  Positive label   : {samples.base_rate:.4f}  (strategy win rate '
        f'{metrics.get("win_rate", 0) / 100:.4f})',
        f'  Features         : {len(samples.feature_names)}',
        sep,
        f'  {_sample_size_note(len(samples))}',
        sep,
    ]))

    if len(samples):
        per_symbol = {}
        for sym in sorted(set(samples.symbol)):
            m = samples.symbol == sym
            per_symbol[sym] = {'n': int(m.sum()), 'win_rate': float(samples.y[m].mean())}
        for sym, s in per_symbol.items():
            logger.info('  %-5s n=%4d  positive=%.4f', sym, s['n'], s['win_rate'])

    if len(samples) < MIN_SAMPLES:
        sys.exit(1)


# ---------------------------------------------------------------------
# walkforward
# ---------------------------------------------------------------------

def _fit_folds(samples, folds, embargo, *, kind, keep_rate, seed, shuffle=False):
    """One estimator + threshold per fold, trained under the purge rule."""
    rng = np.random.default_rng(seed)
    fitted = []
    for train, valid in folds:
        mask = purged_train_mask(samples, train, valid, embargo)
        train_set = samples.subset(mask)
        if len(train_set) < 50 or len(np.unique(train_set.y)) < 2:
            logger.warning(
                '%s: only %d usable training trades; fold skipped (gate will pass through).',
                train.name, len(train_set),
            )
            continue
        y = rng.permutation(train_set.y) if shuffle else train_set.y
        est = fit_estimator(train_set.X, y, train_set.w, kind=kind, seed=seed)
        thr = threshold_for(est, train_set.X, keep_rate)
        logger.debug('%s: %d training trades (positive %.3f), threshold %.4f',
                     train.name, len(train_set), float(train_set.y.mean()), thr)
        fitted.append((valid, est, thr, len(train_set)))
    return fitted


def _bar_date(market, i: int):
    """Bar ``i``'s calendar date, matching what ``BarContext.date`` reports."""
    import pandas as pd

    return pd.Timestamp(market.dates[i]).date()


def _baseline_metrics(market, strategy_cls, params, folds, *, cash, slippage, pad) -> dict:
    """The unfiltered arm on each validation window, computed once.

    It does not depend on ``kind`` or ``keep_rate``, so running it inside the
    settings loop would repeat the same backtest a dozen times for identical
    numbers -- and the settings sweep is already the expensive part.
    """
    base_cls = unfiltered(strategy_cls)
    out = {}
    for _train, valid in folds:
        out[valid.name] = run_window(
            market, base_cls, params, valid, cash=cash, slippage=slippage, pad=pad,
        )['metrics']
    return out


def _evaluate_keep_rate(
    market, strategy_cls, params, samples, folds, baselines, *, keep_rate, kind,
    embargo, cash, slippage, pad, seed, shuffle=False,
):
    """Fit every fold, then compare base vs filtered on each validation window."""
    fitted = _fit_folds(samples, folds, embargo, kind=kind, keep_rate=keep_rate,
                        seed=seed, shuffle=shuffle)
    if not fitted:
        return None

    # Fold boundaries go to the model as dates, not bar indices: `run_window`
    # evaluates each fold on a padded *slice*, whose bar 0 is not the market's
    # bar 0, so an index-addressed model would match nothing there.
    model = WalkForwardMetaModel([
        (_bar_date(market, v.start), _bar_date(market, v.end - 1), est, thr)
        for v, est, thr, _ in fitted
    ])
    filtered_cls = make_meta_filtered(strategy_cls, model)

    fold_reports = []
    for valid, est, thr, n_train in fitted:
        oos = samples.subset(window_mask(samples, valid))
        clf = (classifier_report(oos.y, est.predict_proba(oos.X)[:, 1], thr)
               if len(oos) else {'n': 0, 'auc': float('nan'), 'base_rate': float('nan'),
                                 'keep_rate': float('nan'), 'n_kept': 0,
                                 'precision': float('nan'), 'precision_lift': float('nan'),
                                 'threshold': float(thr)})

        filt_m = run_window(
            market, filtered_cls, params, valid, cash=cash, slippage=slippage, pad=pad,
        )['metrics']
        fold_reports.append({
            'fold': valid.name,
            'bars': [valid.start, valid.end],
            'n_train': n_train,
            'classifier': clf,
            'metrics': compare_metrics(baselines[valid.name], filt_m),
        })

    return {'keep_rate': keep_rate, 'kind': kind, 'folds': fold_reports,
            'summary': summarize_folds(fold_reports)}


def cmd_walkforward(args) -> None:
    strategy_cls = load_strategy(args.strategy)
    params = resolve_params(strategy_cls, args)
    market = load_market(args.symbols, args.start, args.end, update=args.update_data)

    wrapped = unfiltered(strategy_cls)
    pad = probe_warmup(market, wrapped, params)
    logger.info('Warmup pad (features + strategy indicators): %d bars', pad)

    folds, holdout = anchored_walk_forward(
        market.n_bars, reserve_bars=pad, n_folds=args.n_folds,
        embargo=args.embargo, holdout_frac=args.holdout_frac,
    )
    logger.info('Folds: %s | holdout %d-%d (locked)',
                ', '.join(f'{t.name}[{t.start}:{t.end}]/{v.name}[{v.start}:{v.end}]'
                          for t, v in folds), holdout.start, holdout.end)

    # Harvest only over the pre-holdout span so no training trade can come
    # from the locked window, even by accident.
    pre = slice_market(market, 0, holdout.start)
    h = _harvest(pre, strategy_cls, params, cash=args.cash, slippage=args.slippage)
    samples = h['samples']
    logger.info('%s Positive rate %.4f.', _sample_size_note(len(samples)), samples.base_rate)
    if len(samples) < MIN_SAMPLES:
        logger.error('Not enough samples to run a walk-forward. Stopping.')
        sys.exit(1)

    logger.info('Baseline (unfiltered) on each validation window ...')
    baselines = _baseline_metrics(market, strategy_cls, params, folds,
                                  cash=args.cash, slippage=args.slippage, pad=pad)

    results = []
    for kind in args.kind:
        for keep_rate in args.keep_rate:
            logger.info('--- kind=%s keep_rate=%.2f ---', kind, keep_rate)
            r = _evaluate_keep_rate(
                market, strategy_cls, params, samples, folds, baselines,
                keep_rate=keep_rate, kind=kind, embargo=args.embargo,
                cash=args.cash, slippage=args.slippage, pad=pad, seed=args.seed,
            )
            if r is None:
                logger.warning('No fold could be fitted for kind=%s keep_rate=%.2f', kind, keep_rate)
                continue
            s = r['summary']
            logger.info(
                '  AUC %.3f | precision lift %+.4f | keep %.3f | Sharpe delta %+.4f '
                '(%d/%d folds improved) | trades %+.1f',
                s['mean_auc'], s['mean_precision_lift'], s['mean_keep_rate'],
                s['mean_sharpe_delta'], s['folds_sharpe_improved'], s['n_folds'],
                s['mean_trade_delta'],
            )
            results.append(r)

    shuffled = None
    if args.shuffle_control and results:
        target = min(args.keep_rate, key=lambda k: abs(k - 0.65))
        logger.info('--- shuffled-label control (kind=%s keep_rate=%.2f) ---', args.kind[0], target)
        sh = _evaluate_keep_rate(
            market, strategy_cls, params, samples, folds, baselines,
            keep_rate=target, kind=args.kind[0], embargo=args.embargo,
            cash=args.cash, slippage=args.slippage, pad=pad, seed=args.seed, shuffle=True,
        )
        if sh is not None:
            shuffled = sh['summary']
            logger.info('  AUC %.3f | Sharpe delta %+.4f  <- this is the "trading less" effect',
                        shuffled['mean_auc'], shuffled['mean_sharpe_delta'])

    _check_noop_arm(results)

    best = _pick_best(results)
    sep = '=' * 62
    logger.info('\n%s\n  Walk-forward verdict\n%s', sep, sep)
    if best is not None:
        logger.info('  Best non-trivial setting: kind=%s keep_rate=%.2f',
                    best['kind'], best['keep_rate'])
        for fold in best['folds']:
            logger.info('%s', format_table(f"{fold['fold']} (train n={fold['n_train']})",
                                            fold['metrics']))
        logger.info('  %s', verdict(best['summary'], shuffled))
    else:
        logger.info('  No non-trivial setting produced a fitted model.')
    logger.info(sep)

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(args.results_dir, f'{strategy_cls.__name__}_{ts}_walkforward.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'strategy': strategy_cls.__name__, 'strategy_lookup': args.strategy,
            'params': params, 'seed': args.seed,
            'symbols': args.symbols, 'start': args.start, 'end': args.end,
            'cash': args.cash, 'slippage': args.slippage,
            'n_folds': args.n_folds, 'embargo': args.embargo,
            'holdout_frac': args.holdout_frac, 'holdout_bars': [holdout.start, holdout.end],
            'pad': pad, 'n_samples': len(samples), 'base_rate': samples.base_rate,
            'results': results, 'shuffled_control': shuffled,
            'verdict': verdict(best['summary'], shuffled) if best else 'no model fitted',
        }, f, indent=2, default=str)
    logger.info('Report: %s', path)


def _check_noop_arm(results) -> None:
    """``keep_rate=1.0`` keeps every entry, so its arm must match the baseline
    on every fold. Anything else means the harness is not measuring what it
    claims -- either the gate is firing when it should not, or (the failure
    this check was written for) it is *not* firing when it should, and every
    other keep_rate is silently reporting the baseline too.
    """
    for r in results:
        if r['keep_rate'] < 1.0:
            continue
        for fold in r['folds']:
            for key, row in fold['metrics'].items():
                if abs(row['delta']) > 1e-9:
                    logger.warning(
                        'SELF-CHECK FAILED: keep_rate=1.0 (%s) changed %s on %s by %+g. '
                        'The no-op arm must reproduce the baseline exactly; treat every '
                        'number in this report as unreliable until that is explained.',
                        r['kind'], key, fold['fold'], row['delta'],
                    )
                    return
    if any(r['keep_rate'] >= 1.0 for r in results):
        logger.info('Self-check passed: keep_rate=1.0 reproduces the baseline exactly.')


def _pick_best(results):
    """The best setting among the non-trivial ones (keep_rate < 1.0).

    ``keep_rate=1.0`` is the self-check arm -- it must reproduce the baseline
    exactly -- so it is never a candidate for "best". Ranked on mean Sharpe
    delta, which is what the shuffled control is then compared against.

    This is a selection over every (kind, keep_rate) pair, so the winner's
    Sharpe delta is optimistic by construction. That is why ``verdict`` leads
    with AUC and precision lift -- neither of which this choice can inflate --
    and why it ends by sending the reader to the holdout.
    """
    candidates = [r for r in results if r['keep_rate'] < 1.0]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (
        r['summary']['mean_sharpe_delta']
        if np.isfinite(r['summary']['mean_sharpe_delta']) else -np.inf
    ))


# ---------------------------------------------------------------------
# holdout
# ---------------------------------------------------------------------

def cmd_holdout(args) -> None:
    """Evaluate one setting on the locked window -- once.

    Mirrors ``research.optimize.evaluate_holdout``: the window is used at most
    once and the report records that it has been. Two extra guards, because
    the meta layer adds two extra ways to spend it badly:

    * a report whose verdict is NO EDGE / NOT PROVEN is refused, since
      confirming a filter the walk-forward already rejected is just buying a
      second opinion until one agrees;
    * ``--keep-rate`` must be given explicitly, so the setting is a decision
      rather than whatever happened to top the walk-forward table.
    """
    with open(args.report, encoding='utf-8') as f:
        report = json.load(f)

    if report.get('holdout_evaluated') and not args.force:
        logger.error(
            'This report already recorded a holdout evaluation on %s. The window is '
            'meant to be used once; re-running it turns it into another validation '
            'set. Pass --force if you accept that.', report['holdout_evaluated'],
        )
        sys.exit(1)

    said = report.get('verdict', '')
    if not args.force and not said.startswith('SIGNAL'):
        logger.error(
            'The walk-forward verdict on this report was:\n  %s\n'
            'Spending the locked window on a filter that already failed out of sample '
            'only asks a second judge for a different answer. Pass --force to override.',
            said,
        )
        sys.exit(1)

    strategy_cls = load_strategy(args.strategy or report['strategy_lookup'])
    params = report['params']
    market = load_market(report['symbols'], report['start'], report['end'])
    holdout = Window('holdout', *report['holdout_bars'])
    pad, embargo = report['pad'], report['embargo']

    # Train on everything that closed before the embargo barrier, so nothing
    # from the holdout -- or from the embargo gap in front of it -- is seen.
    pre = slice_market(market, 0, holdout.start)
    samples = _harvest(pre, strategy_cls, params,
                       cash=report['cash'], slippage=report['slippage'])['samples']
    barrier = holdout.start - embargo
    train = samples.subset(samples.close_bar < barrier)
    logger.info('Training on %d trades closing before bar %d (%s).',
                len(train), barrier, _bar_date(market, barrier))

    est = fit_estimator(train.X, train.y, train.w, kind=args.kind, seed=report.get('seed', 42))
    thr = threshold_for(est, train.X, args.keep_rate)
    model = WalkForwardMetaModel([
        (_bar_date(market, holdout.start), _bar_date(market, holdout.end - 1), est, thr),
    ])

    base_m = run_window(market, unfiltered(strategy_cls), params, holdout,
                        cash=report['cash'], slippage=report['slippage'], pad=pad)['metrics']
    filt_m = run_window(market, make_meta_filtered(strategy_cls, model), params, holdout,
                        cash=report['cash'], slippage=report['slippage'], pad=pad)['metrics']
    comparison = compare_metrics(base_m, filt_m)

    oos = samples.subset(window_mask(samples, holdout))
    sep = '=' * 62
    logger.info('\n%s\n  Holdout: %s -> %s  (used once)\n%s', sep,
                _bar_date(market, holdout.start), _bar_date(market, holdout.end - 1), sep)
    logger.info('%s', format_table(f'kind={args.kind} keep_rate={args.keep_rate:.2f}', comparison))
    logger.info(sep)

    report['holdout_evaluated'] = datetime.datetime.now().isoformat(timespec='seconds')
    report['holdout_result'] = {
        'kind': args.kind, 'keep_rate': args.keep_rate, 'threshold': thr,
        'n_train': len(train), 'n_holdout_samples': len(oos), 'metrics': comparison,
    }
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info('Recorded in %s', args.report)


# ---------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------

def cmd_fit(args) -> None:
    strategy_cls = load_strategy(args.strategy)
    params = resolve_params(strategy_cls, args)
    market = load_market(args.symbols, args.start, args.end, update=args.update_data)

    h = _harvest(market, strategy_cls, params, cash=args.cash, slippage=args.slippage)
    samples = h['samples']
    logger.info('%s Positive rate %.4f.', _sample_size_note(len(samples)), samples.base_rate)
    if len(samples) < MIN_SAMPLES:
        sys.exit(1)

    est = fit_estimator(samples.X, samples.y, samples.w, kind=args.kind, seed=args.seed)
    thr = threshold_for(est, samples.X, args.keep_rate)
    trained_through = str(market.dates[-1])

    model = FittedMetaModel(
        estimator=est, threshold_value=thr, keep_rate=args.keep_rate,
        kind=args.kind, trained_through=trained_through,
    )
    save_model(model, args.out, meta={
        'strategy': args.strategy, 'strategy_class': strategy_cls.__name__,
        'params': params, 'symbols': args.symbols,
        'train_start': args.start, 'train_end': args.end,
        'cash': args.cash, 'slippage': args.slippage,
        'n_samples': len(samples), 'base_rate': samples.base_rate,
    })

    in_sample = (est.predict_proba(samples.X)[:, 1] >= thr)
    logger.info(
        'Fitted on %d trades through %s. Threshold %.4f keeps %.1f%% in sample.\n'
        'These in-sample numbers are not evidence -- run `walkforward` for that.',
        len(samples), trained_through, thr, 100.0 * in_sample.mean(),
    )
    print(f'Next: python meta_runner.py signal --model {args.out}'
          f'  (same thing: python live_runner.py --model {args.out})')


# ---------------------------------------------------------------------
# signal -- an alias for `live_runner.py --model`
# ---------------------------------------------------------------------

def cmd_signal(args) -> None:
    """Today's filtered target positions.

    Kept in this CLI because ``fit`` -> ``signal`` reads better as one
    sequence, but the work itself lives in ``live/``: replaying to the last
    bar and reading ``engine.pending`` is not a meta-labeling idea, and the
    same code serves an ungated strategy from ``strategies/`` (see
    ``live_runner.py``). This subcommand is exactly
    ``live_runner.py --model <path>``.
    """
    spec = spec_from_model(
        args.model, symbols=args.symbols, cash=args.cash, slippage=args.slippage,
    )
    market = load_market(spec.symbols, args.start, args.end, update=args.update_data)
    emit(compute_signal(market, spec), args.results_dir)


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

def _add_data_args(p, *, from_artifact=False):
    """Shared data/strategy flags.

    ``from_artifact`` is the ``signal`` case: the strategy, its params, cash
    and slippage all come out of the saved model, so those flags become
    optional overrides that default to "whatever was trained with" rather than
    to a fresh value that would silently disagree with the artifact.
    """
    if not from_artifact:
        p.add_argument('--strategy', required=True, choices=_strategy_choices())
        p.add_argument('--cash', type=float, default=1_000_000.0)
        p.add_argument('--slippage', type=float, default=1.0)
        p.add_argument('--lots', type=int, default=None)
        p.add_argument('--params-from', default=None,
                        help="Load strategy params from an optimize '*_best.json' report.")
        p.add_argument('--param', action='append', metavar='NAME=VALUE',
                        help='Override one strategy param; repeatable.')
    else:
        p.add_argument('--cash', type=float, default=None,
                        help='Override the training cash. Set this to your REAL account '
                             'equity for strategies that size off equity.')
        p.add_argument('--slippage', type=float, default=None)
    p.add_argument('--symbols', nargs='+', default=None if from_artifact else list(DEFAULT_SYMBOLS),
                    help=f'Products to load (registered: {", ".join(list_products())})')
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default='2026-12-31')
    p.add_argument('--update-data', action='store_true')


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FuturesToolkit meta-labeling CLI.')
    parser.add_argument('--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('harvest', help='Run the primary unfiltered and report the sample set.')
    _add_data_args(p)
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser('walkforward', help='Purged walk-forward: does the filter add anything?')
    _add_data_args(p)
    p.add_argument('--n-folds', type=int, default=4)
    p.add_argument('--embargo', type=int, default=10)
    p.add_argument('--holdout-frac', type=float, default=0.20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep-rate', nargs='+', type=float, default=list(DEFAULT_KEEP_RATES),
                    help='Fractions of entries to keep. 1.0 is the no-op self-check.')
    p.add_argument('--kind', nargs='+', default=['rf', 'lr'], choices=['rf', 'lr'])
    p.add_argument('--shuffle-control', action='store_true',
                    help='Refit on permuted labels to measure the trade-less-often effect.')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser('holdout', help='Evaluate one setting on the locked window (once).')
    p.add_argument('--report', required=True, help='A walkforward *_walkforward.json report.')
    p.add_argument('--keep-rate', type=float, required=True,
                    help='Must be stated explicitly, so the setting is a decision rather '
                         'than whatever topped the walk-forward table.')
    p.add_argument('--kind', default='rf', choices=['rf', 'lr'])
    p.add_argument('--strategy', default=None, choices=_strategy_choices(),
                    help='Override the strategy recorded in the report.')
    p.add_argument('--force', action='store_true',
                    help='Evaluate anyway: re-use a spent window, or override a NO EDGE verdict.')
    p.set_defaults(func=cmd_holdout)

    p = sub.add_parser('fit', help='Fit a final model on all history and save it.')
    _add_data_args(p)
    p.add_argument('--keep-rate', type=float, default=0.65)
    p.add_argument('--kind', default='rf', choices=['rf', 'lr'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('-o', '--out', default=os.path.join(ROOT_DIR, 'models', 'meta.joblib'))
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser('signal', help="Today's filtered target positions "
                                      "(alias for `live_runner.py --model`).")
    _add_data_args(p, from_artifact=True)
    p.add_argument('--model', required=True, help='Path to a `fit` artifact.')
    p.add_argument('--results-dir', default=LIVE_RESULTS_DIR)
    p.set_defaults(func=cmd_signal)

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    try:
        args.func(args)
    except (ValueError, RuntimeError, AssertionError, FileNotFoundError) as e:
        logger.error('%s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
