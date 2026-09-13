"""
FuturesToolkit CLI -- the single entry point for the whole toolkit.

Usage (from the repo root):
  python ft.py data SA CF RB                     # download / rebuild exchange history
  python ft.py backtest --strategy double_ma     # run a backtest, write charts + trade log
  python ft.py show-space --strategy double_ma   # print a strategy's parameter ranges
  python ft.py validate --strategy double_ma     # overfitting checks on those parameters
  python ft.py web                               # serve the browser panel

Run ``python ft.py <subcommand> --help`` for each command's full flag list.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import logging
import math
import os
import re
import socket
import sys

from datafeed.products import list_products, require_products
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
    print(f'{strategy_cls.__name__} parameter ranges:')
    for name, spec in space.items():
        print(f'  {name}: {spec}')
    fixed = set(getattr(strategy_cls, 'fixed_params', ()) or ())
    unscanned = [k for k in (strategy_cls.params or {}) if k not in space and k not in fixed]
    if fixed:
        print(f'  (fixed, never perturbed: {sorted(fixed)})')
    if unscanned:
        print(f'  (no range could be inferred, left at default: {unscanned})')


# ---------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------

def _fmt(value, spec: str = '.4f', width: int = 8) -> str:
    """Format a number that may legitimately be NaN or None.

    Every check can decline to answer -- a window too short to bootstrap, a
    dimension whose neighbours the strategy's constraints rule out -- and
    printing 0.0000 for those would claim a result the tool does not have.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{'n/a':>{width}}"
    return f'{value:>{width}{spec}}'


def _print_walkforward(check: dict) -> list:
    lines = ['  Walk-forward consistency', '  ' + '-' * 56,
             f"  {'window':<12}{'sharpe':>9}{'trades':>8}{'maxDD%':>9}{'score':>9}"]
    for row in check['train'] + check['valid']:
        lines.append(
            f"  {row['name']:<12}{_fmt(row['sharpe_ratio'], '.2f', 9)}"
            f"{row['n_trades']:>8}{_fmt(row['max_drawdown'], '.1f', 9)}"
            f"{_fmt(row['score'], '.2f', 9)}"
        )
    decay = check['decay']
    lines.append(
        f"  mean IS {_fmt(decay['is_score'], '.2f', 6)}   mean OOS "
        f"{_fmt(decay['oos_score'], '.2f', 6)}   ratio {_fmt(decay['ratio'], '.2f', 6)}"
    )
    return lines


def _print_subperiod(check: dict) -> list:
    lines = ['  Sub-period stability', '  ' + '-' * 56,
             f"  {'period':<12}{'sharpe':>9}{'trades':>8}{'maxDD%':>9}{'return%':>10}"]
    for row in check['periods']:
        lines.append(
            f"  {row['name']:<12}{_fmt(row['sharpe_ratio'], '.2f', 9)}"
            f"{row['n_trades']:>8}{_fmt(row['max_drawdown'], '.1f', 9)}"
            f"{_fmt(row['total_return'], '.1f', 10)}"
        )
    worst = check['worst_period']
    if worst:
        lines.append(
            f"  worst: {worst['name']} at sharpe {worst['sharpe_ratio']:.2f}   "
            f"positive periods: {check['positive_fraction'] * check['n_periods']:.0f}"
            f"/{check['n_periods']}"
        )
    return lines


def _print_sensitivity(check: dict) -> list:
    lines = ['  Parameter sensitivity', '  ' + '-' * 56,
             f"  {'param':<24}{'max drop%':>12}{'verdict':>14}"]
    for name, dim in sorted(check['dimensions'].items()):
        if not dim.get('evaluated'):
            verdict = 'not measured'
        elif dim['flags_spike']:
            verdict = 'SPIKE'
        elif dim['max_drop_pct'] < -10.0:
            # A negative drop means a neighbour scored *better*, so this value
            # is not even the local best. Worth saying plainly rather than
            # printing a minus sign next to the word "flat": it is the
            # opposite of the failure this check looks for.
            verdict = 'off-peak'
        else:
            verdict = 'flat'
        drop = dim['max_drop_pct'] if dim.get('evaluated') else None
        lines.append(f"  {name:<24}{_fmt(drop, '.1f', 12)}{verdict:>14}")
    lines.append(f"  base score {check['base_score']:.4f}")
    if check['unmeasured']:
        lines.append(
            f"  {len(check['unmeasured'])} dimension(s) had no measurable neighbour: "
            f"{', '.join(check['unmeasured'])}"
        )
    return lines


def _print_bootstrap(check: dict) -> list:
    lines = ['  Block bootstrap', '  ' + '-' * 56]
    if check['insufficient']:
        lines.append(f"  Not run: {check['n_obs']} bars is too short for block={check['block']}.")
        return lines
    sharpe, dd = check['sharpe'], check['max_drawdown']
    lines += [
        f"  {check['n_draws']} draws, block={check['block']} bars",
        f"  sharpe        point {_fmt(sharpe['point'], '.2f', 7)}",
        f"                95% CI[{_fmt(sharpe['ci_low'], '.2f', 7)},{_fmt(sharpe['ci_high'], '.2f', 7)} ]",
        f"                P(SR>0){_fmt(sharpe['p_positive'], '.3f', 7)}",
        f"  max drawdown  observed {_fmt(dd['observed'], '.1f', 6)}%   "
        f"median {_fmt(dd['median'], '.1f', 6)}%   95th {_fmt(dd['p95'], '.1f', 6)}%",
    ]
    return lines


def _print_contribution(metrics: dict) -> list:
    """Net P&L per product, straight off the full-sample run.

    Not one of the checks and not flagged -- every check above reads the
    portfolio equity curve, which is one axis: time. This is the other axis,
    and it is free to print because ``compute_metrics`` already computed it.
    A "multi-product" strategy whose total is one product's P&L with noise
    stapled to it passes every time-axis check there is, so the numbers have
    to at least be visible.
    """
    by_symbol = metrics.get('by_symbol') or {}
    if not by_symbol:
        return []
    rows = sorted(by_symbol.items(), key=lambda kv: kv[1]['net_pnl'], reverse=True)
    total = sum(abs(stats['net_pnl']) for _, stats in rows) or 1.0
    lines = ['  Per-product contribution (not a check)', '  ' + '-' * 56,
             f"  {'product':<10}{'trades':>8}{'net P&L':>16}{'share':>9}"]
    for symbol, stats in rows:
        lines.append(
            f"  {symbol:<10}{stats['n_trades']:>8}{stats['net_pnl']:>16,.0f}"
            f"{abs(stats['net_pnl']) / total * 100:>8.0f}%"
        )
    losers = [s for s, st in rows if st['net_pnl'] < 0]
    if losers:
        lines.append(f"  Losing products: {', '.join(losers)}")
    return lines


def _print_selection(checks: dict) -> list:
    lines = ['  Selection-bias corrections', '  ' + '-' * 56]
    if 'pbo' in checks:
        pbo = checks['pbo']
        lines.append(
            f"  PBO (CSCV)        {_fmt(pbo['pbo'], '.3f', 7)}   over "
            f"{pbo['n_combinations']} split(s) of {pbo['n_blocks']} block(s)"
        )
    if 'dsr' in checks:
        dsr = checks['dsr']
        declared = 'declared' if dsr.get('trials_declared') else 'grid size'
        lines += [
            f"  Deflated Sharpe   {_fmt(dsr['dsr'], '.3f', 7)}   "
            # Per *bar*, and said so: these are the raw ratios the deflation
            # works on, an annualization factor away from every other Sharpe
            # in this report.
            f"sr_hat {_fmt(dsr.get('sr_hat'), '.3f', 6)}  sr0 "
            f"{_fmt(dsr.get('sr0'), '.3f', 6)}  (per bar)",
            f"                    n_trials={dsr.get('n_trials')} ({declared}), "
            f"candidates={dsr.get('n_candidates')}",
        ]
    if 'pbo' in checks:
        lines += [
            '  Candidates are a one-step neighbourhood, not a search, so PBO reads',
            '  narrower than the same statistic over independently tried configurations.',
        ]
    return lines


_VERDICTS = {
    'walkforward': 'out-of-sample folds keep less than half the in-sample score',
    'subperiod': 'the worst sub-period loses money and under half of them are positive',
    'sensitivity': 'at least one parameter is a lone spike rather than a plateau',
    'bootstrap': 'resampling cannot rule out a true Sharpe of zero at 95%',
    'pbo': 'an in-sample pick among the neighbours usually fails out of sample',
    'dsr': 'the Sharpe does not survive correction for how many configurations were tried',
}


def _print_validation(report: dict, path: str) -> None:
    sep = '=' * 60
    lines = [sep, f"  Overfitting Report -- {report['strategy']}", sep,
             f"  Products  : {', '.join(report['symbols'])}",
             f"  Params    : {report['params']}",
             f"  Window    : bars [{report['evaluation_window']['start']}, "
             f"{report['evaluation_window']['end']}) of {report['n_bars']}",
             f"  Full-sample sharpe {report['full_metrics'].get('sharpe_ratio', 0):.4f}  "
             f"trades {report['full_metrics'].get('n_trades', 0)}  "
             f"maxDD {report['full_metrics'].get('max_drawdown', 0):.2f}%",
             sep]

    checks = report['checks']
    printers = (
        ('walkforward', _print_walkforward), ('subperiod', _print_subperiod),
        ('sensitivity', _print_sensitivity), ('bootstrap', _print_bootstrap),
    )
    for key, printer in printers:
        if key in checks:
            lines += printer(checks[key]) + [sep]
    if 'pbo' in checks or 'dsr' in checks:
        lines += _print_selection(checks) + [sep]
    contribution = _print_contribution(report['full_metrics'])
    if contribution:
        lines += contribution + [sep]

    flags = report['flags']
    if flags:
        lines.append(f'  *** {len(flags)} CHECK(S) FLAGGED ***')
        for name in flags:
            lines.append(f'  - {name}: {_VERDICTS[name]}')
    else:
        lines.append('  No check flagged. That is not a pass -- it is the absence of')
        lines.append('  these particular failures on this particular history.')
    if report['skipped']:
        lines.append(f"  Skipped: {', '.join(report['skipped'])}")
    lines += [sep, f'  Report: {path}', sep]
    logger.info('\n'.join(lines))


def cmd_validate(args) -> dict:
    from research.space import resolve_params
    from research.validate import validate_strategy

    strategy_cls = _load_strategy(args.strategy)
    symbols = require_products(args.symbols)
    params = resolve_params(strategy_cls, args)

    out = validate_strategy(
        strategy_cls=strategy_cls,
        symbols=symbols,
        params=params,
        start=args.start,
        end=args.end,
        cash=args.cash,
        slippage=args.slippage,
        n_folds=args.n_folds,
        embargo=args.embargo,
        holdout_frac=args.holdout_frac,
        n_periods=args.n_periods,
        trials_tried=args.trials_tried,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_block=args.bootstrap_block,
        seed=args.seed,
        skip=tuple(args.skip or ()),
        results_dir=args.results_dir,
        update_data=args.update_data,
    )
    _print_validation(out['report'], out['path'])
    print(f"Replay it with the full trade log and charts: "
          f"python ft.py backtest --strategy {args.strategy} --params-from {out['path']}")

    if args.fail_on_warn and out['report']['flags']:
        sys.exit(1)
    return out


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
    p.add_argument('--n-folds', type=int, default=4,
                    help='Anchored walk-forward folds (default: %(default)s)')
    p.add_argument('--embargo', type=int, default=10,
                    help='Bars dropped between each train window and its valid window')
    # 0.0, not the 0.20 the deleted search used. Holding a window back protects
    # it from *selection*, and nothing here selects: the parameters arrive
    # already chosen. Reserving a tail would only shorten the folds and hide
    # the most recent stretch, which is the one worth seeing.
    p.add_argument('--holdout-frac', type=float, default=0.0,
                    help='Trailing fraction excluded from the folds entirely '
                         '(default: %(default)s -- the folds span all history)')
    p.add_argument('--seed', type=int, default=42)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='ft.py', description='FuturesToolkit: data, backtesting, and overfitting checks.',
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
                    help="Load strategy params from an `ft.py validate` "
                         "'*_validation.json' report, so a checked parameter set can be "
                         'run here with the full trade log and charts.')
    p.add_argument('--param', action='append', metavar='NAME=VALUE',
                    help='Override one strategy param; repeatable. Beats --params-from.')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    p.add_argument('--keep-last', type=int, default=None,
                    help='Delete result files from all but the N most recent runs')
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser('show-space', help="Print the ranges a strategy's params are scanned over.")
    p.add_argument('--strategy', required=True, metavar='NAME',
                    help=f'A discovered name ({", ".join(_strategy_names())}) '
                         f"or a 'module.path:ClassName' reference")
    p.set_defaults(func=cmd_show_space)

    from research.validate import CHECKS

    p = sub.add_parser(
        'validate',
        help='Check one parameter set for overfitting: walk-forward, sub-period, '
             'sensitivity, bootstrap, PBO and Deflated Sharpe.',
    )
    p.add_argument('--strategy', required=True, metavar='NAME',
                    help=f'A discovered name ({", ".join(_strategy_names())}) '
                         f"or a 'module.path:ClassName' reference")
    _add_data_args(p)
    _add_split_args(p)
    p.add_argument('--param', action='append', metavar='NAME=VALUE',
                    help='The parameter value to check; repeatable. Defaults to the '
                         "strategy's own declared default.")
    p.add_argument('--params-from', default=None,
                    help="Read params from an earlier `ft.py validate` report.")
    p.add_argument('--lots', type=int, default=None,
                    help='Lots per trade (strategies that use it); default 1')
    p.add_argument('--trials-tried', type=int, default=None,
                    help='How many parameter sets you tried before settling on this one. '
                         "Feeds the Deflated Sharpe's multiple-testing correction, which "
                         'otherwise assumes only the neighbourhood grid was ever looked '
                         'at and so reads optimistically for a hand-tuned strategy.')
    p.add_argument('--n-periods', type=int, default=0,
                    help='Sub-periods to split the sample into; 0 (default) splits by '
                         'calendar year')
    p.add_argument('--bootstrap-draws', type=int, default=2000)
    p.add_argument('--bootstrap-block', type=int, default=20,
                    help='Bootstrap block length in bars; set it at least as long as a '
                         'typical holding period (default: %(default)s)')
    p.add_argument('--skip', action='append', choices=list(CHECKS), default=None,
                    help='Leave a check out; repeatable')
    p.add_argument('--fail-on-warn', action='store_true',
                    help='Exit non-zero when any check flags, for use in a gate')
    p.add_argument('--results-dir', default=None)
    p.set_defaults(func=cmd_validate)

    from web.config import DEFAULT_HOST, DEFAULT_PORT

    p = sub.add_parser('web', help='Serve the browser panel (products, data, backtest, history).')
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
