"""
FuturesToolkit backtest runner.

Usage (from the repo root):
  python runner.py --symbols SA CF RB AG C --start 2020-01-01 --end 2026-12-31 \
      --strategy double_ma --cash 100000

To replay a tuned parameter set from ``research_runner.py optimize``, point
``--params-from`` at its ``*_best.json`` report.

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

from core.backtest import run_single_backtest
from core.market import build_market_data
from core.ledger import TRADE_LOG_FIELDS
from datafeed.data_manager import DataManager
from datafeed.products import list_products, require_products
from plotting import BacktestPlotter
from strategies import discover_strategies, load_strategy

# Re-exported: this used to be defined here, and `meta_runner.py` and the
# tests still reach it by this name.
__all__ = ['run_single_backtest', 'parse_param_value', 'resolve_params', 'main']

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DEFAULT_SYMBOLS = [
    'SA', 'FG', 'CF',   # CZCE
    'C',                # DCE
]
DEFAULT_START = '2020-01-01'
DEFAULT_END = '2026-12-31'
DEFAULT_CASH = 100_000.0
DEFAULT_SLIPPAGE = 0.0
DEFAULT_STRATEGY = 'double_ma'

STRATEGIES = discover_strategies()

logger = logging.getLogger('futurestoolkit.runner')


def parse_param_value(raw: str) -> tuple:
    """Parse one ``name=value`` CLI token into ``(name, value)``, casting the
    value to int, then float, then bool, else leaving it a string.

    Distinct from ``optimize``'s ``--param name=kind:args``, which declares a
    search *range* -- see ``research.space.parse_param_override``.
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
    parser.add_argument('--symbols', nargs='+', default=list(DEFAULT_SYMBOLS),
                         help=f'Products to load (default: {" ".join(DEFAULT_SYMBOLS)}; '
                              f'registered: {", ".join(list_products())})')
    parser.add_argument('--start', default=DEFAULT_START, help="Backtest start date 'YYYY-MM-DD'")
    parser.add_argument('--end', default=DEFAULT_END, help="Backtest end date 'YYYY-MM-DD'")
    parser.add_argument('--cash', type=float, default=DEFAULT_CASH, help='Initial cash (CNY)')
    parser.add_argument('--strategy', choices=sorted(STRATEGIES), default=DEFAULT_STRATEGY,
                         help='Strategy to run')
    parser.add_argument('--slippage', type=float, default=DEFAULT_SLIPPAGE,
                         help='Fill slippage in ticks, scaled per product by '
                              "products.py's tick_size")
    parser.add_argument('--lots', type=int, default=None,
                         help='Lots per trade (strategies that use it); default 1')
    parser.add_argument('--params-from', default=None,
                         help="Load strategy params from a research_runner.py optimize "
                              "'*_best.json' report, so a tuned parameter set can be run "
                              'here with the full trade log and charts.')
    parser.add_argument('--param', action='append', metavar='NAME=VALUE',
                         help='Override one strategy param; repeatable. Beats --params-from.')
    parser.add_argument('--meta-model', default=None,
                         help='Path to a meta_runner.py `fit` artifact. Wraps --strategy so '
                              'the model vetoes low-probability entries (exits are never '
                              'blocked). The run is in-sample wherever the model was trained; '
                              'use `meta_runner.py walkforward` to measure the filter.')
    parser.add_argument('--update-data', action='store_true', help='Refresh exchange data before running')
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
    if not metrics:
        # Every field below would read 0.00 -- including `Initial Cash`, which
        # the user demonstrably did set. Say what happened instead; the engine
        # has already logged the specific reason just above this.
        logger.warning('\n'.join([
            sep, '  Backtest Result Summary', sep,
            '  No bars were recorded, so every metric is undefined.',
            '  See the engine warning above for the reason (most often the',
            "  strategy's indicator warmup exceeds the loaded date range).",
            f'  Trade Log           : {log_path}', sep,
        ]))
        return
    lines = [sep, '  Backtest Result Summary', sep]
    if metrics.get('blown_up'):
        # Loud, and above the numbers: every figure below is a truncated window.
        lines += [
            '  *** ACCOUNT BLOWN UP -- run stopped early ***',
            '  Equity hit zero; the bars after that were never traded.',
            sep,
        ]
    lines += [
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
        f"  Rejected Orders     : {metrics.get('n_rejected_orders', 0):>14}",
        sep,
    ]
    for symbol, stats in metrics.get('by_symbol', {}).items():
        lines.append(
            f"  {symbol:5} n={stats['n_trades']:3d}  win_rate={stats['win_rate']:6.2f}%  "
            f"net_pnl={stats['net_pnl']:>12,.2f}  expectancy={stats['expectancy']:>10.2f}"
        )
    if metrics.get('by_symbol'):
        lines.append(sep)
    for reason, stats in sorted(metrics.get('by_exit_reason', {}).items()):
        lines.append(
            f"  exit={reason:<12} n={stats['n_trades']:3d} ({stats['share']:5.1f}%)  "
            f"win_rate={stats['win_rate']:6.2f}%  "
            f"net_pnl={stats['net_pnl']:>12,.2f}  expectancy={stats['expectancy']:>10.2f}"
        )
    if metrics.get('by_exit_reason'):
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
    strategy_cls = load_strategy(args.strategy)
    params = resolve_params(strategy_cls, args)

    if args.meta_model:
        from meta.filter import make_meta_filtered
        from meta.model import load_model

        meta_model = load_model(args.meta_model)
        strategy_cls = make_meta_filtered(strategy_cls, meta_model)
    strategy_name = strategy_cls.__name__

    logger.info('=' * 60)
    logger.info('  FuturesToolkit Backtest')
    logger.info('  Products : %s', ', '.join(symbols))
    logger.info('  Strategy : %s  [%s -> %s]', strategy_name, args.start, args.end)
    logger.info('  Slippage : %s', 'off' if not args.slippage else f'{args.slippage:g} ticks')
    if args.meta_model:
        logger.info('  Meta     : %s (keep_rate %.2f, threshold %.4f)',
                    os.path.basename(args.meta_model), meta_model.keep_rate,
                    meta_model.threshold_value)
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
