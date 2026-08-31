"""
FuturesToolkit backtest runner.

Usage (from the repo root):
  python runner.py --symbols SA FG CF --start 2020-01-01 --end 2026-12-31 \
      --strategy double_ma --cash 100000

Run ``python runner.py --help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import logging
import os
import re

from core.engine import Engine
from core.market import MarketData, build_market_data
from core.metrics import compute_metrics
from core.ledger import TRADE_LOG_FIELDS
from datafeed.data_manager import DataManager
from datafeed.products import list_products, require_products
from plotting import BacktestPlotter
from strategies import discover_strategies
from strategies.base import BarContext, SetupContext

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

STRATEGIES = discover_strategies()

logger = logging.getLogger('futurestoolkit.runner')


def run_single_backtest(
    market: MarketData,
    strategy_cls: type,
    params: dict,
    cash: float,
    slippage: float = 0.0,
    warmup_bars: int = 0,
    bar_context_cls: type = BarContext,
) -> dict:
    """Run one backtest with no side effects (no plotting, no file writes).

    Reused by both ``main()`` below and ``research_runner.py`` (parameter
    optimization runs this once per trial per fold; meta-labeling runs it
    once per walk-forward fold, with and without the gating wrapper).
    ``bar_context_cls`` defaults to the normal ``BarContext`` but can be
    swapped for a wrapper (e.g. ``research.gating``'s recording/gating
    contexts) that needs the exact same warmup/pad plumbing.
    """
    strategy = strategy_cls(**params)
    engine = Engine(market, strategy, initial_cash=cash, slippage=slippage, warmup_bars=warmup_bars)
    result = engine.run_backtest(SetupContext, bar_context_cls)
    metrics = compute_metrics(
        result['equity_records'], result['trade_logs'], cash,
        liquidation_count=result['liquidation_count'],
    )
    return {'result': result, 'metrics': metrics, 'engine': engine}


def parse_param_value(raw: str) -> tuple:
    """Parse one ``name=value`` CLI token into ``(name, value)``, casting the
    value to int, then float, then bool, else leaving it a string.

    Shared with ``research_runner.py`` (its ``metalabel`` / ``meta-backtest``
    commands take the same form) so a param spelled one way on one CLI means
    the same thing on the other. Distinct from ``optimize``'s ``--param
    name=kind:args``, which declares a search *range* -- see
    ``research.space.parse_param_override``.
    """
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


def resolve_params(strategy_cls: type, args: argparse.Namespace) -> dict:
    """Build the ``strategy_cls(**overrides)`` dict from the CLI, lowest
    precedence first: an optimize report's ``best_params``, then ``--lots``,
    then ``--param``. Only what the user actually asked to change is returned
    -- ``Strategy.__init__`` merges the class's own ``params`` defaults under
    it -- so an untouched run behaves exactly as before.
    """
    params: dict = {}
    if getattr(args, 'params_from', None):
        with open(args.params_from, encoding='utf-8') as f:
            best = json.load(f)['best_params']
        params.update(best)
        logger.info('Params from %s: %s', args.params_from, best)
    if getattr(args, 'lots', None) is not None:
        params['lots'] = args.lots
    for raw in getattr(args, 'param', None) or []:
        name, value = parse_param_value(raw)
        params[name] = value

    known = set(getattr(strategy_cls, 'params', {}) or {})
    unknown = sorted(set(params) - known)
    if unknown:
        # Not fatal: `Strategy.__init__` merges anything into `self.p`, and
        # `--lots` is documented as harmless for strategies that ignore it.
        # But a typo would otherwise vanish without a trace, so say so.
        logger.warning(
            '%s declares no param(s) %s -- passing them through, but the strategy '
            'will not read them (declared params: %s).',
            strategy_cls.__name__, unknown, sorted(known) or '<none>',
        )
    return params


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FuturesToolkit backtest runner.')
    parser.add_argument('--symbols', nargs='+', default=['SA', 'FG', 'CF'],
                         help=f'Products to load (default: SA FG CF; registered: {", ".join(list_products())})')
    parser.add_argument('--start', default='2020-01-01', help="Backtest start date 'YYYY-MM-DD'")
    parser.add_argument('--end', default='2026-12-31', help="Backtest end date 'YYYY-MM-DD'")
    parser.add_argument('--cash', type=float, default=100_000.0, help='Initial cash (CNY)')
    parser.add_argument('--strategy', choices=sorted(STRATEGIES), default='double_ma')
    parser.add_argument('--slippage', type=float, default=0.0, help='Fill slippage in price points')
    parser.add_argument('--lots', type=int, default=None,
                         help='Lots per trade (strategies that use it); default 1')
    parser.add_argument('--params-from', default=None,
                         help="Load strategy params from a research_runner.py optimize "
                              "'*_best.json' report, so a tuned parameter set can be run "
                              'here with the full trade log and charts.')
    parser.add_argument('--param', action='append', metavar='NAME=VALUE',
                         help='Override one strategy param; repeatable. Beats --params-from.')
    parser.add_argument('--update-data', action='store_true', help='Refresh CZCE data before running')
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--keep-last', type=int, default=None,
                         help='Delete result files from all but the N most recent runs')
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--quiet', action='store_true')
    verbosity.add_argument('--verbose', action='store_true')
    return parser.parse_args(argv)


def _configure_logging(args: argparse.Namespace) -> None:
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format='%(message)s')


def _save_trade_log(trade_logs: list, results_dir: str, strategy_name: str) -> str:
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(results_dir, f'{strategy_name}_trades_{ts}.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(trade_logs)
    logger.info('Trade log saved: %s', path)
    return path


def _print_summary(metrics: dict, log_path: str) -> None:
    sep = '=' * 50
    lines = [
        sep, '  Backtest Result Summary', sep,
        f"  Initial Cash        : {metrics.get('initial_cash', 0):>14,.2f} CNY",
        f"  Final Equity        : {metrics.get('final_equity', 0):>14,.2f} CNY",
        f"  Total Return        : {metrics.get('total_return', 0):>14.4f} %",
        f"  Annualized Return   : {metrics.get('annualized_return', 0):>14.4f} %",
        f"  Annualized Vol      : {metrics.get('annualized_volatility', 0):>14.4f} %",
        f"  Sharpe Ratio        : {metrics.get('sharpe_ratio', 0):>14.4f}",
        f"  Sortino Ratio       : {metrics.get('sortino_ratio', 0):>14.4f}",
        f"  Max Drawdown        : {metrics.get('max_drawdown', 0):>14.4f} %",
    ]
    recovery_days = metrics.get('max_drawdown_recovery_days')
    if recovery_days is not None:
        recovery_text = f'{recovery_days:>14} trading days'
    else:
        recovery_text = f"{'Not recovered':>14}"
    calmar = metrics.get('calmar_ratio', 0)
    calmar_text = f'{calmar:.4f}' if calmar != float('inf') else 'inf'
    lines += [
        f"  MaxDD Recovery      : {recovery_text}",
        f"  Calmar Ratio        : {calmar_text:>14}",
        f"  Win Rate            : {metrics.get('win_rate', 0):>14.4f} %",
        f"  Profit/Loss Ratio   : {metrics.get('profit_loss_ratio', 0):>14.4f}",
        f"  Expectancy          : {metrics.get('expectancy', 0):>14.2f} CNY/trade",
    ]
    pf = metrics.get('profit_factor', 0)
    pf_text = f'{pf:.4f}' if pf != float('inf') else 'inf'
    lines += [
        f"  Profit Factor       : {pf_text:>14}",
        f"  Avg Holding Days    : {metrics.get('avg_holding_days', 0):>14.2f}",
        f"  Capital Exposure    : {metrics.get('capital_exposure', 0):>14.4f} %",
        f"  Turnover            : {metrics.get('turnover', 0):>14.4f} x",
        f"  Total Trades        : {metrics.get('n_trades', 0):>14}",
        f"  Winning Trades      : {metrics.get('n_winning', 0):>14}",
        f"  Losing Trades       : {metrics.get('n_losing', 0):>14}",
        f"  Forced Liquidations : {metrics.get('n_forced_liquidations', 0):>14}",
        sep,
    ]
    for symbol, stats in metrics.get('by_symbol', {}).items():
        lines.append(
            f"  {symbol:5} n={stats['n_trades']:3d}  win_rate={stats['win_rate']:6.2f}%  "
            f"net_pnl={stats['net_pnl']:>12,.2f}  expectancy={stats['expectancy']:>10.2f}"
        )
    if metrics.get('by_symbol'):
        lines.append(sep)
    lines += [f'  Trade Log           : {log_path}', sep]
    logger.info('\n'.join(lines))


_TS_RE = re.compile(r'_(\d{8}_\d{6})\.')


def _cleanup_old_results(results_dir: str, keep_last: int) -> None:
    if keep_last is None or keep_last < 0:
        return
    timestamps = set()
    for path in glob.glob(os.path.join(results_dir, '*')):
        m = _TS_RE.search(os.path.basename(path))
        if m:
            timestamps.add(m.group(1))
    stale = sorted(timestamps, reverse=True)[keep_last:]
    if not stale:
        return
    removed = 0
    for path in glob.glob(os.path.join(results_dir, '*')):
        m = _TS_RE.search(os.path.basename(path))
        if m and m.group(1) in stale:
            os.remove(path)
            removed += 1
    logger.info('Removed %d file(s) from %d older run(s) in %s', removed, len(stale), results_dir)


def main(argv=None) -> dict:
    args = _parse_args(argv)
    _configure_logging(args)

    symbols = require_products(args.symbols)
    strategy_cls = STRATEGIES[args.strategy]
    strategy_name = strategy_cls.__name__
    params = resolve_params(strategy_cls, args)

    logger.info('=' * 60)
    logger.info('  FuturesToolkit Backtest')
    logger.info('  Products : %s', ', '.join(symbols))
    logger.info('  Strategy : %s  [%s -> %s]', strategy_name, args.start, args.end)
    logger.info('  Slippage : %s', 'off' if not args.slippage else f'{args.slippage:g} price points')
    logger.info('=' * 60)

    logger.info('[1/3] Loading data ...')
    dm = DataManager(symbols=symbols, update=args.update_data)
    universe = dm.get_universe_bundle(start_date=args.start, end_date=args.end)
    market = build_market_data(universe)

    logger.info('[2/3] Running backtest ...')
    outcome = run_single_backtest(market, strategy_cls, params, args.cash, args.slippage)
    result, metrics = outcome['result'], outcome['metrics']

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = _save_trade_log(result['trade_logs'], args.results_dir, strategy_name)
    _print_summary(metrics, log_path)

    logger.info('[3/3] Plotting charts ...')
    price_dfs = {sym: universe['products'][sym]['weighted_df'] for sym in symbols}
    exec_price_dfs = {sym: universe['products'][sym]['exec_price_df'] for sym in symbols}
    plotter = BacktestPlotter(
        equity_records=result['equity_records'],
        trade_logs=result['trade_logs'],
        price_dfs=price_dfs,
        signal_log=result['signal_log'],
        metrics=metrics,
        config={'results_dir': args.results_dir, 'strategy_name': strategy_name},
        exec_price_dfs=exec_price_dfs,
        symbols=symbols,
    )
    chart_paths = plotter.plot_all()

    if args.keep_last is not None:
        _cleanup_old_results(args.results_dir, args.keep_last)

    logger.info('=' * 60)
    logger.info('  Chart file paths')
    logger.info('=' * 60)
    for key, path in chart_paths.items():
        logger.info('  %-9s: %s', key, path)
    logger.info('  trade_log: %s', log_path)
    logger.info('=' * 60)

    return {'result': result, 'metrics': metrics, 'chart_paths': chart_paths, 'log_path': log_path}


if __name__ == '__main__':
    main()
