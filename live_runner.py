"""
FuturesToolkit live signal CLI: the next session's target positions.

Works for any strategy ``strategies.discover_strategies()`` finds, with or
without a meta-model:

  # A strategy straight out of strategies/
  python live_runner.py --strategy double_ma --update-data

  # ... replaying a tuned parameter set from `research_runner.py optimize`
  python live_runner.py --strategy double_ma \
      --params-from results/optuna/DoubleMaStrategy_<ts>_best.json

  # A meta-labeled strategy: the primary, its params and the training
  # universe all come out of the `meta_runner.py fit` artifact
  python live_runner.py --model models/dma.joblib --cash 500000

``--strategy`` and ``--model`` are the two mutually exclusive ways of saying
what to run. Everything after that is shared, because everything after that
is the same computation -- see ``live/signal.py``.

Run ``python live_runner.py --help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from datafeed.products import list_products, require_products
from research.runner_api import load_market
from runner import resolve_params
from strategies import discover_strategies, load_strategy

from live.report import emit
from live.signal import SignalSpec, compute_signal, spec_from_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'live')
DEFAULT_SYMBOLS = ['SA', 'FG', 'CF', 'C']
DEFAULT_START = '2015-01-01'
DEFAULT_END = '2026-12-31'
DEFAULT_CASH = 1_000_000.0
DEFAULT_SLIPPAGE = 1.0

# Flags that describe the *primary* strategy. A `fit` artifact already fixed
# every one of them, and honouring them alongside --model would score today's
# bar with a model trained on a different strategy -- wrong, but silently so.
_PRIMARY_ONLY = ('strategy', 'params_from', 'param', 'lots')

logger = logging.getLogger('futurestoolkit.live')


def build_spec(args: argparse.Namespace) -> SignalSpec:
    """Resolve the CLI into a :class:`SignalSpec`, from whichever source."""
    if args.model:
        conflicting = [f'--{n.replace("_", "-")}' for n in _PRIMARY_ONLY if getattr(args, n, None)]
        if conflicting:
            raise ValueError(
                f'{", ".join(conflicting)} cannot be combined with --model: the fit '
                'artifact already fixes the strategy and its params, and a model '
                'scored against a different primary is silently wrong. Re-run '
                '`meta_runner.py fit` if the primary changed.'
            )
        spec = spec_from_model(
            args.model,
            symbols=args.symbols, cash=args.cash, slippage=args.slippage,
        )
        return SignalSpec(**{**vars(spec), 'symbols': require_products(spec.symbols)})

    strategy_cls = load_strategy(args.strategy)
    return SignalSpec(
        strategy_cls=strategy_cls,
        params=resolve_params(strategy_cls, args),
        symbols=require_products(args.symbols or DEFAULT_SYMBOLS),
        cash=DEFAULT_CASH if args.cash is None else args.cash,
        slippage=DEFAULT_SLIPPAGE if args.slippage is None else args.slippage,
    )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='FuturesToolkit live signal: the next session\'s target positions.',
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument('--strategy', choices=sorted(discover_strategies()),
                      help='A strategy from strategies/, run ungated.')
    what.add_argument('--model', default=None,
                      help='A `meta_runner.py fit` artifact: runs the primary it was '
                           'trained on, with entries gated by the model.')

    parser.add_argument('--symbols', nargs='+', default=None,
                        help=f'Products to load (default: the training universe with '
                             f'--model, else {" ".join(DEFAULT_SYMBOLS)}; '
                             f'registered: {", ".join(list_products())})')
    parser.add_argument('--start', default=DEFAULT_START,
                        help='Start of the replay. It sets the simulated position and '
                             'equity the signal is computed from, so keep it stable '
                             'between runs.')
    parser.add_argument('--end', default=DEFAULT_END)
    parser.add_argument('--cash', type=float, default=None,
                        help='Set this to your REAL account equity for strategies that '
                             f'size off equity (default: the training cash with --model, '
                             f'else {DEFAULT_CASH:,.0f}).')
    parser.add_argument('--slippage', type=float, default=None,
                        help='Fill slippage in ticks; default is the training run\'s.')
    parser.add_argument('--lots', type=int, default=None, help='--strategy only.')
    parser.add_argument('--params-from', default=None,
                        help="--strategy only: params from an optimize '*_best.json'.")
    parser.add_argument('--param', action='append', metavar='NAME=VALUE',
                        help='--strategy only: override one param; repeatable.')
    parser.add_argument('--update-data', action='store_true',
                        help='Refresh exchange data first. Without it the signal is as '
                             'of the last bar already on disk, which may not be today.')
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(message)s')
    try:
        spec = build_spec(args)
        market = load_market(spec.symbols, args.start, args.end, update=args.update_data)
        report = compute_signal(market, spec)
    except (ValueError, RuntimeError, AssertionError, KeyError, FileNotFoundError) as e:
        logger.error('%s', e)
        sys.exit(1)
    emit(report, args.results_dir)
    return report


if __name__ == '__main__':
    main()
