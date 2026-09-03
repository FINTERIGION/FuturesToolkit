"""Performance metrics computed from ``equity_records`` + ``trade_logs``.

Ported from the old ``BacktestEngine._calc_metrics`` (backtrader era), plus
the additions from docs/rewrite-plan.md §8: annualized return/vol, Sortino,
Calmar, average holding days, capital exposure, turnover, a per-symbol
breakdown, and forced-liquidation count. The Sharpe/Sortino annualization
factor is derived from the run's actual bars-per-year instead of a hardcoded
252 (the sample here runs closer to ~244 trading days/year).
"""

from __future__ import annotations

import logging
import math
from datetime import date as Date
from typing import Dict, List

import numpy as np

from datafeed.products import product_costs

logger = logging.getLogger(__name__)

_DEFAULT_TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    equity_records: List[dict],
    trade_logs: List[dict],
    initial_cash: float,
    liquidation_count: int = 0,
    rejected_count: int = 0,
    blown_up: bool = False,
    risk_free_rate: float = 0.03,
) -> dict:
    if not equity_records:
        return {}

    final_equity = equity_records[-1]['equity']
    total_return = (final_equity - initial_cash) / initial_cash

    dr = np.array([r['daily_return'] for r in equity_records], dtype=float)
    equities = np.array([r['equity'] for r in equity_records], dtype=float)
    dates = [r['date'] for r in equity_records]

    trading_days_per_year = _annualization_factor(dates)
    risk_free_daily = risk_free_rate / trading_days_per_year
    excess = dr - risk_free_daily

    sharpe = (
        excess.mean() / excess.std() * math.sqrt(trading_days_per_year)
        if excess.std() > 1e-10 else 0.0
    )
    downside = excess[excess < 0]
    sortino = (
        excess.mean() / downside.std() * math.sqrt(trading_days_per_year)
        if len(downside) > 0 and downside.std() > 1e-10 else 0.0
    )

    # Max drawdown. Once the running peak itself is non-positive, drawdown as
    # a fraction of peak is undefined -- treat that region as a full (100%)
    # drawdown instead of dividing by a zero/negative peak.
    running_max = np.maximum.accumulate(equities)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (running_max - equities) / running_max
    drawdowns = np.where(running_max > 0, ratio, 1.0)
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0

    if max_drawdown <= 1e-12:
        max_drawdown_recovery_days = 0
    else:
        trough_idx = int(np.argmax(drawdowns))
        peak_equity = running_max[trough_idx]
        peak_hits = np.flatnonzero(equities[:trough_idx + 1] == peak_equity)
        peak_idx = int(peak_hits[-1]) if len(peak_hits) else trough_idx
        recovery_indices = np.flatnonzero(equities[trough_idx + 1:] >= peak_equity) + trough_idx + 1
        max_drawdown_recovery_days = (
            int(recovery_indices[0] - peak_idx) if len(recovery_indices) else None
        )

    n_years = max(len(dates) / trading_days_per_year, 1e-9)
    annualized_return = (
        (1.0 + total_return) ** (1.0 / n_years) - 1.0 if total_return > -1 else -1.0
    )
    annualized_volatility = float(dr.std() * math.sqrt(trading_days_per_year))
    calmar_ratio = (
        annualized_return / max_drawdown if max_drawdown > 1e-9 else float('inf')
    )

    n_trades = len(trade_logs)
    if n_trades > 0:
        winning = [t for t in trade_logs if t['net_pnl'] > 0]
        losing = [t for t in trade_logs if t['net_pnl'] < 0]
        win_rate = len(winning) / n_trades
        avg_win = sum(t['net_pnl'] for t in winning) / len(winning) if winning else 0.0
        avg_loss = abs(sum(t['net_pnl'] for t in losing) / len(losing)) if losing else 0.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else float('inf')
        expectancy = sum(t['net_pnl'] for t in trade_logs) / n_trades
        gross_wins = sum(t['net_pnl'] for t in winning)
        gross_losses = abs(sum(t['net_pnl'] for t in losing))
        profit_factor = gross_wins / gross_losses if gross_losses > 1e-10 else float('inf')
        holding_days = [
            (t['close_date'] - t['open_date']).days
            for t in trade_logs if t['close_date'] and t['open_date']
        ]
        avg_holding_days = sum(holding_days) / len(holding_days) if holding_days else 0.0
    else:
        winning, losing = [], []
        win_rate = profit_loss_ratio = expectancy = profit_factor = 0.0
        avg_win = avg_loss = avg_holding_days = 0.0

    turnover = _turnover(trade_logs, equities)
    capital_exposure = _capital_exposure(equity_records)
    by_symbol = _per_symbol_breakdown(trade_logs)
    by_exit_reason = _exit_reason_breakdown(trade_logs)

    # Books check: every yuan the equity curve moved must be accounted for by
    # a logged trade. Fails if the equity curve picks up a term the trade log
    # doesn't know about, or if trades are being split/double-counted.
    booked = sum(t['net_pnl'] for t in trade_logs)
    drift = booked - (final_equity - initial_cash)
    tol = max(1e-6 * abs(initial_cash), 0.01)
    if abs(drift) > tol:
        logger.warning(
            "Trade log does not reconcile with the equity curve: "
            "sum(net_pnl)=%.2f vs equity change=%.2f (drift %.2f)",
            booked, final_equity - initial_cash, drift,
        )

    return {
        'initial_cash': initial_cash,
        'final_equity': round(final_equity, 2),
        'total_return': round(total_return * 100, 4),
        'annualized_return': round(annualized_return * 100, 4),
        'annualized_volatility': round(annualized_volatility * 100, 4),
        'sharpe_ratio': round(sharpe, 4),
        'sortino_ratio': round(sortino, 4),
        'max_drawdown': round(max_drawdown * 100, 4),
        'max_drawdown_recovery_days': max_drawdown_recovery_days,
        'calmar_ratio': round(calmar_ratio, 4) if calmar_ratio != float('inf') else calmar_ratio,
        'win_rate': round(win_rate * 100, 4),
        'profit_loss_ratio': (
            round(profit_loss_ratio, 4) if profit_loss_ratio != float('inf') else profit_loss_ratio
        ),
        'expectancy': round(expectancy, 4),
        'profit_factor': round(profit_factor, 4) if profit_factor != float('inf') else profit_factor,
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'avg_holding_days': round(avg_holding_days, 2),
        'capital_exposure': round(capital_exposure * 100, 4),
        'turnover': round(turnover, 4),
        'n_trades': n_trades,
        'n_winning': len(winning),
        'n_losing': len(losing),
        'n_forced_liquidations': liquidation_count,
        'n_rejected_orders': rejected_count,
        # True when the run stopped early on an insolvent account: every metric
        # here then describes a truncated window, not the requested one.
        'blown_up': blown_up,
        'reconciliation_drift': round(drift, 4),
        'by_symbol': by_symbol,
        'by_exit_reason': by_exit_reason,
    }


def _annualization_factor(dates: List[Date]) -> float:
    if len(dates) < 2:
        return _DEFAULT_TRADING_DAYS_PER_YEAR
    span_days = (dates[-1] - dates[0]).days
    if span_days <= 0:
        return _DEFAULT_TRADING_DAYS_PER_YEAR
    years = span_days / 365.25
    if years <= 0:
        return _DEFAULT_TRADING_DAYS_PER_YEAR
    return len(dates) / years


def _capital_exposure(equity_records: List[dict]) -> float:
    ratios = [
        r['margin_used'] / r['equity']
        for r in equity_records
        if r.get('equity') and r['equity'] > 0 and 'margin_used' in r
    ]
    return sum(ratios) / len(ratios) if ratios else 0.0


def _turnover(trade_logs: List[dict], equities: np.ndarray) -> float:
    if not trade_logs or len(equities) == 0:
        return 0.0
    avg_equity = float(np.mean(equities))
    if avg_equity <= 0:
        return 0.0
    total_notional = 0.0
    for t in trade_logs:
        try:
            mult = product_costs(t['symbol'])['multiplier']
        except KeyError:
            continue
        total_notional += (t['open_price'] + t['close_price']) * t['size'] * mult
    return total_notional / avg_equity


def _exit_reason_breakdown(trade_logs: List[dict]) -> Dict[str, dict]:
    """How each kind of exit paid: the dial for tuning a stop/target bracket.

    A bracket's two levels trade win rate against expectancy, and the totals
    hide that -- widening a target lifts the average winner and thins the
    winners at the same time, and only the split by exit says which of the two
    moved. ``share`` is the percentage of all trades that left this way.
    """
    by_reason: Dict[str, List[dict]] = {}
    for t in trade_logs:
        by_reason.setdefault(t.get('exit_reason') or 'unknown', []).append(t)
    total = len(trade_logs)
    out = {}
    for reason, trades in by_reason.items():
        n = len(trades)
        wins = [t for t in trades if t['net_pnl'] > 0]
        out[reason] = {
            'n_trades': n,
            'share': round(n / total * 100, 4) if total else 0.0,
            'win_rate': round(len(wins) / n * 100, 4) if n else 0.0,
            'net_pnl': round(sum(t['net_pnl'] for t in trades), 4),
            'expectancy': round(sum(t['net_pnl'] for t in trades) / n, 4) if n else 0.0,
        }
    return out


def _per_symbol_breakdown(trade_logs: List[dict]) -> Dict[str, dict]:
    by_symbol: Dict[str, List[dict]] = {}
    for t in trade_logs:
        by_symbol.setdefault(t['symbol'], []).append(t)
    out = {}
    for symbol, trades in by_symbol.items():
        n = len(trades)
        wins = [t for t in trades if t['net_pnl'] > 0]
        out[symbol] = {
            'n_trades': n,
            'win_rate': round(len(wins) / n * 100, 4) if n else 0.0,
            'net_pnl': round(sum(t['net_pnl'] for t in trades), 4),
            'expectancy': round(sum(t['net_pnl'] for t in trades) / n, 4) if n else 0.0,
        }
    return out
