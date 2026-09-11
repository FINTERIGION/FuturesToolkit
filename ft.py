"""
FuturesToolkit CLI -- the single entry point for the whole toolkit.

Usage (from the repo root):
  python ft.py data SA CF RB                     # download / rebuild exchange history
  python ft.py backtest --strategy double_ma     # run a backtest, write charts + trade log
  python ft.py show-space --strategy double_ma   # print a strategy's tunable space
  python ft.py optimize --strategy double_ma     # Optuna walk-forward parameter search
  python ft.py holdout --best results/optuna/..._best.json
  python ft.py web                               # serve the browser panel

Run ``python ft.py <subcommand> --help`` for each command's full flag list.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import logging
import os
import re
import socket
import sys

from datafeed.products import list_products, require_products
from research.objective import DEFAULT_SPARSE_PENALTY
from strategies import discover_strategies, load_strategy

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DEFAULT_SYMBOLS = [
    'SA', 'FG', 'CF',   # CZCE
    'C',                # DCE
]
DEFAULT_START = '2020-01-01'
DEFAULT_END = '2026-12-31'
DEFAULT_CASH = 200_000.0
DEFAULT_SLIPPAGE = 0.0
DEFAULT_STRATEGY = 'double_ma'

logger = logging.getLogger('futurestoolkit.cli')


def _strategy_names() -> list:
    return sorted(discover_strategies())


def _load_strategy(spec: str) -> type:
    """``load_strategy``, with its out-of-tree failure modes folded into the
    one error type ``main`` already reports cleanly.

    ``--strategy`` is deliberately not an argparse ``choices=`` list. That
    rejected every ``'module.path:ClassName'`` spec before ``load_strategy``
    ever saw it -- the form its own docstring documents, and the only way to
    run a strategy that lives outside this repo. Validating here instead
    keeps both forms working and still names the discovered ones on a typo,
    because that is exactly what ``load_strategy``'s ``KeyError`` says.
    """
    try:
        return load_strategy(spec)
    except KeyError as e:
        # `str(KeyError)` is the *repr* of its argument, so letting this reach
        # `main`'s `logger.error('%s', e)` prints the whole sentence wrapped in
        # stray quotes. Re-raise as the type that formats plainly.
        raise ValueError(e.args[0]) from e
    except (ImportError, AttributeError, TypeError) as e:
        raise ValueError(
            f"Could not load strategy {spec!r}: {e}. Expected a name from "
            f"{_strategy_names()} or a 'module.path:ClassName' reference."
        ) from e


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------

def cmd_data(args) -> None:
    from datafeed.data_update import run_updates

    run_updates(args.symbols, force=args.force, rebuild_only=args.rebuild_only)


# ---------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------

def _save_trade_log(trade_logs: list, results_dir: str, strategy_name: str) -> str:
    from core.ledger import TRADE_LOG_FIELDS

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


def cmd_backtest(args) -> dict:
    from core.backtest import run_single_backtest
    from core.market import build_market_data
    from datafeed.data_manager import DataManager
    from plotting import BacktestPlotter
    from research.space import resolve_params

    symbols = require_products(args.symbols)
    strategy_cls = _load_strategy(args.strategy)
    params = resolve_params(strategy_cls, args)
    strategy_name = strategy_cls.__name__

    logger.info('=' * 60)
    logger.info('  FuturesToolkit Backtest')
    logger.info('  Products : %s', ', '.join(symbols))
    logger.info('  Strategy : %s  [%s -> %s]', strategy_name, args.start, args.end)
    logger.info('  Slippage : %s', 'off' if not args.slippage else f'{args.slippage:g} ticks')
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


# ---------------------------------------------------------------------
# show-space
# ---------------------------------------------------------------------

def cmd_show_space(args) -> None:
    from research.space import resolve_space

    strategy_cls = _load_strategy(args.strategy)
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

    strategy_cls = _load_strategy(args.strategy)
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
        sparse_penalty=args.sparse_penalty,
        param_overrides=overrides or None,
        seed=args.seed,
        probe_samples=args.probe_samples,
        study_name=args.study_name,
        results_dir=args.results_dir,
        update_data=args.update_data,
    )
    # `run_study` already logs the report path at INFO; only the follow-up
    # hint is this layer's to print.
    print(f"Next: python ft.py holdout --best {out['path']}")
    print(f"Or replay it with the full trade log and charts: "
          f"python ft.py backtest --strategy {args.strategy} --params-from {out['path']}")


# ---------------------------------------------------------------------
# holdout
# ---------------------------------------------------------------------

def cmd_holdout(args) -> None:
    from research.optimize import evaluate_holdout

    evaluate_holdout(args.best, force=args.force)


# ---------------------------------------------------------------------
# web
# ---------------------------------------------------------------------

_LOOPBACK_BINDS = ('127.0.0.1', 'localhost', '::1')
_WILDCARD_BINDS = ('0.0.0.0', '::', '*', '')


def _bracketed(address: str) -> str:
    """An IPv6 literal in the form a Host header carries it."""
    return f'[{address}]' if ':' in address and not address.startswith('[') else address


def _local_addresses() -> list:
    """This machine's own addresses, as another device on the LAN sees them.

    A UDP socket is *connected* to a documentation-range address to make the
    kernel pick an outbound route; no packet is sent and nothing has to be
    reachable. That is the one way to learn the address that actually carries
    traffic off this box -- ``gethostbyname`` answers 127.0.1.1 on a good many
    Linux installs, which is exactly the address a remote browser will not be
    using.
    """
    found = []
    for family, probe in ((socket.AF_INET, ('192.0.2.1', 9)), (socket.AF_INET6, ('2001:db8::1', 9))):
        try:
            sock = socket.socket(family, socket.SOCK_DGRAM)
            try:
                sock.connect(probe)
                found.append(_bracketed(sock.getsockname()[0]))
            finally:
                sock.close()
        except OSError:
            continue    # no route on this family; nothing to add
    return found


def _panel_allowed_hosts(bind_host: str) -> list:
    """Host names to accept when the panel is bound to ``bind_host``.

    This used to be ``'*'`` -- the check turned *off* for exactly the bind
    that exposes the panel to other machines, which is the one where it is
    load-bearing. Fail-open on a security control, and it left the panel
    answering to any hostname an attacker cared to point at it.

    Deriving the list instead costs nothing in the common cases. A concrete
    bind address is the Host a browser will send, verbatim. A wildcard bind
    cannot be read off the flag, so the machine's own addresses and hostname
    stand in for it. Neither admits a name an attacker controls, which is
    what a rebinding page needs.
    """
    host = (bind_host or '').strip()
    if host.lower() not in _WILDCARD_BINDS:
        return [_bracketed(host)]

    from web.config import DEFAULT_ALLOWED_HOSTS

    hosts = list(DEFAULT_ALLOWED_HOSTS) + _local_addresses()
    try:
        machine = socket.gethostname()
    except OSError:
        machine = ''
    if machine and machine not in hosts:
        hosts.append(machine)
    return hosts


def cmd_web(args) -> None:
    import uvicorn

    from web.config import ALLOWED_HOSTS_ENV

    if args.host not in _LOOPBACK_BINDS:
        # The panel refuses a Host header it does not recognise, which is what
        # stops a web page the user has open from driving this API through
        # their browser. On loopback the defaults cover it; bound anywhere else
        # the Host is whatever name the operator reaches the box by, so it is
        # derived from the bind address rather than stood down.
        if not os.environ.get(ALLOWED_HOSTS_ENV):
            derived = _panel_allowed_hosts(args.host)
            os.environ[ALLOWED_HOSTS_ENV] = ','.join(derived)
            host_note = (
                f' Answering to {", ".join(derived)} only; if you reach it by some '
                f'other name, set {ALLOWED_HOSTS_ENV} to a comma-separated list of '
                'the hostnames you serve it under.'
            )
        else:
            host_note = f' Host header restricted to {ALLOWED_HOSTS_ENV}.'
        logger.warning(
            'Binding to %s: this panel has no authentication and can rewrite the '
            'product registry and delete data files. Only do this on a network '
            'you trust.%s', args.host, host_note,
        )
    uvicorn.run('web.app:app', host=args.host, port=args.port, reload=args.reload)


# ---------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------

def _add_data_args(p) -> None:
    p.add_argument('--symbols', nargs='+', default=list(DEFAULT_SYMBOLS),
                    help=f'Products to load (default: {" ".join(DEFAULT_SYMBOLS)}; '
                         f'registered: {", ".join(list_products())})')
    p.add_argument('--start', default=DEFAULT_START, help="Start date 'YYYY-MM-DD'")
    p.add_argument('--end', default=DEFAULT_END, help="End date 'YYYY-MM-DD'")
    p.add_argument('--cash', type=float, default=DEFAULT_CASH, help='Initial cash (CNY)')
    p.add_argument('--slippage', type=float, default=DEFAULT_SLIPPAGE,
                    help="Fill slippage in ticks, scaled per product by products.py's tick_size")
    p.add_argument('--update-data', action='store_true',
                    help='Refresh exchange data before running')


def _add_split_args(p) -> None:
    p.add_argument('--n-folds', type=int, default=4)
    p.add_argument('--embargo', type=int, default=10)
    p.add_argument('--holdout-frac', type=float, default=0.20)
    p.add_argument('--seed', type=int, default=42)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='ft.py', description='FuturesToolkit: data, backtesting, and parameter tuning.',
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--quiet', action='store_true')
    verbosity.add_argument('--verbose', action='store_true')
    sub = parser.add_subparsers(dest='command', required=True)

    from datafeed.data_update import add_cli_args as add_data_update_args

    p = sub.add_parser('data', help='Download exchange history and build OI-weighted daily bars.')
    add_data_update_args(p)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser('backtest', help='Run one backtest; write charts and a trade log.')
    p.add_argument('--strategy', default=DEFAULT_STRATEGY, metavar='NAME',
                    help=f'Strategy to run: a discovered name ({", ".join(_strategy_names())}) '
                         f"or a 'module.path:ClassName' reference (default: %(default)s)")
    _add_data_args(p)
    p.add_argument('--lots', type=int, default=None,
                    help='Lots per trade (strategies that use it); default 1')
    p.add_argument('--params-from', default=None,
                    help="Load strategy params from an `ft.py optimize` '*_best.json' report, "
                         'so a tuned parameter set can be run here with the full trade log '
                         'and charts.')
    p.add_argument('--param', action='append', metavar='NAME=VALUE',
                    help='Override one strategy param; repeatable. Beats --params-from.')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    p.add_argument('--keep-last', type=int, default=None,
                    help='Delete result files from all but the N most recent runs')
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser('show-space', help="Print a strategy's tunable search space.")
    p.add_argument('--strategy', required=True, metavar='NAME',
                    help=f'A discovered name ({", ".join(_strategy_names())}) '
                         f"or a 'module.path:ClassName' reference")
    p.set_defaults(func=cmd_show_space)

    p = sub.add_parser('optimize', help='Optuna anchored walk-forward parameter search.')
    p.add_argument('--strategy', required=True, metavar='NAME',
                    help=f'A discovered name ({", ".join(_strategy_names())}) '
                         f"or a 'module.path:ClassName' reference")
    _add_data_args(p)
    _add_split_args(p)
    p.add_argument('--n-trials', type=int, default=200)
    p.add_argument('--probe-samples', type=int, default=20)
    p.add_argument('--lambda-std', type=float, default=0.5)
    p.add_argument('--min-trades-per-year', type=float, default=4.0)
    p.add_argument('--dd-cap', type=float, default=0.35)
    p.add_argument('--sparse-penalty', type=float, default=DEFAULT_SPARSE_PENALTY,
                    help='How hard to mark down a window that produced fewer trades '
                         'than --min-trades-per-year expects, at the extreme of no '
                         'trades at all (default: %(default)s). Raise it toward 1.0 to '
                         'demand the evidence be there before a configuration counts; '
                         'lower it to let a promising-but-thin one keep exploring.')
    p.add_argument('--param', action='append',
                    help='Search-space override: name=kind:args, e.g. slow_period=int:20:200')
    p.add_argument('--study-name', default=None)
    p.add_argument('--results-dir', default=None)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser('holdout', help='Evaluate an optimize report on its locked holdout window (once).')
    p.add_argument('--best', required=True, help='Path to an optimize *_best.json report.')
    p.add_argument('--force', action='store_true', help='Re-evaluate even if already evaluated.')
    p.set_defaults(func=cmd_holdout)

    from web.config import DEFAULT_HOST, DEFAULT_PORT

    p = sub.add_parser('web', help='Serve the browser panel (products, data, backtest, optimize, history).')
    p.add_argument('--host', default=DEFAULT_HOST)
    p.add_argument('--port', type=int, default=DEFAULT_PORT)
    p.add_argument('--reload', action='store_true')
    p.set_defaults(func=cmd_web)

    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=level, format='%(message)s')
    try:
        return args.func(args)
    except (ValueError, RuntimeError, KeyError) as e:
        logger.error('%s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
