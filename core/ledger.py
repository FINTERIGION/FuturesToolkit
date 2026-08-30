"""Fill-driven logical trade ledger.

One row per *logical* trade: the fill that takes a product off flat through
the fill that flattens it again, with every calendar roll in between folded
into that same row (see docs/rewrite-plan.md, migrated from the old
backtrader-analyzer ``TradeLogAnalyzer``).

Realized P&L comes straight from ``Fill.realized_pnl`` (computed by
``core.broker.Broker`` at fill time), not re-derived here -- that keeps the
ledger's books identical to the cash broker actually moved, which is what
makes the reconciliation invariant (``Σ net_pnl == final_equity -
initial_cash``) hold by construction rather than by coincidence.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Dict, List, Optional

from datafeed.products import product_costs

from .broker import Broker
from .types import Fill, Reason

TRADE_LOG_FIELDS = [
    'trade_id', 'open_date', 'close_date', 'direction', 'symbol',
    'contract', 'contracts', 'n_rolls',
    'open_price', 'close_price', 'size',
    'gross_pnl', 'commission', 'net_pnl', 'margin_used',
    'open_at_end', 'forced',
]


class Ledger:
    def __init__(self):
        self.trades: List[dict] = []
        self._open: Dict[str, dict] = {}
        self._net: Dict[str, int] = {}
        self._next_id = 1

    # ------------------------------------------------------------------

    def process_fill(self, fill: Fill) -> None:
        symbol = fill.symbol

        if fill.reason == Reason.ROLL:
            row = self._open.get(symbol)
            if row is not None:
                if fill.contract not in row['contracts']:
                    row['contracts'].append(fill.contract)
                row['roll_days'].add(fill.date)
                row['commission'] += fill.commission
                row['gross_pnl'] += fill.realized_pnl
            return

        prev = self._net.get(symbol, 0)
        new = prev + fill.size
        self._net[symbol] = new
        forced = fill.reason == Reason.LIQUIDATION

        close_qty = min(abs(prev), abs(fill.size))
        total_qty = abs(fill.size)
        open_qty = total_qty - close_qty
        close_comm = fill.commission * close_qty / total_qty if total_qty else 0.0
        open_comm = fill.commission - close_comm

        row = self._open.get(symbol)

        if close_qty:
            if row is None:
                row = self._begin(symbol, fill, prev)  # defensive: shouldn't happen
            row['exit_qty'] += close_qty
            row['exit_notional'] += close_qty * fill.price
            row['gross_pnl'] += fill.realized_pnl
            row['commission'] += close_comm
            row['size'] = max(row['size'], abs(prev))
            costs = product_costs(symbol)
            row['margin_used'] = max(
                row['margin_used'],
                abs(prev) * fill.price * costs['multiplier'] * costs['margin_rate'],
            )
            if fill.contract not in row['contracts']:
                row['contracts'].append(fill.contract)
            if forced:
                row['forced'] = 1
            if new == 0 or (prev > 0) != (new > 0):
                row['close_date'] = fill.date
                self._finalize(symbol, row)
                self._open.pop(symbol, None)
                row = None

        if open_qty:
            if row is None:
                row = self._open[symbol] = self._begin(symbol, fill, new)
            row['entry_qty'] += open_qty
            row['entry_notional'] += open_qty * fill.price
            row['commission'] += open_comm
            row['size'] = max(row['size'], abs(new))
            costs = product_costs(symbol)
            row['margin_used'] = max(
                row['margin_used'],
                abs(new) * fill.price * costs['multiplier'] * costs['margin_rate'],
            )
            if fill.contract not in row['contracts']:
                row['contracts'].append(fill.contract)

    def finish(self, broker: Broker, last_date: Date) -> None:
        """Close out rows still open at the end of the run, at their last mark."""
        for (symbol, contract), pos in list(broker.positions.items()):
            row = self._open.get(symbol)
            if row is None or pos.size == 0:
                continue
            multiplier = product_costs(symbol)['multiplier']
            qty = abs(pos.size)
            row['exit_qty'] += qty
            row['exit_notional'] += qty * pos.last_mark
            row['gross_pnl'] += (pos.last_mark - pos.avg_entry) * pos.size * multiplier
            row['open_at_end'] = 1
            row['close_date'] = last_date
            if contract not in row['contracts']:
                row['contracts'].append(contract)
            self._finalize(symbol, row)
            self._open.pop(symbol, None)

    # ------------------------------------------------------------------

    def _begin(self, symbol: str, fill: Fill, net: int) -> dict:
        tid = self._next_id
        self._next_id += 1
        return {
            'id': tid,
            'symbol': symbol,
            'direction': 'long' if net > 0 else 'short',
            'open_date': fill.date,
            'close_date': None,
            'contract': fill.contract,
            'contracts': [],
            'roll_days': set(),
            'entry_qty': 0, 'entry_notional': 0.0,
            'exit_qty': 0, 'exit_notional': 0.0,
            'size': 0, 'margin_used': 0.0,
            'gross_pnl': 0.0, 'commission': 0.0,
            'open_at_end': 0, 'forced': 0,
        }

    def _finalize(self, symbol: str, row: dict) -> None:
        entry = row['entry_notional'] / row['entry_qty'] if row['entry_qty'] else 0.0
        exitp = row['exit_notional'] / row['exit_qty'] if row['exit_qty'] else entry
        self.trades.append({
            'trade_id': row['id'],
            'open_date': row['open_date'],
            'close_date': row['close_date'],
            'direction': row['direction'],
            'symbol': symbol,
            'contract': row['contract'],
            'contracts': '|'.join(row['contracts']),
            'n_rolls': len(row['roll_days']),
            'open_price': round(entry, 4),
            'close_price': round(exitp, 4),
            'size': row['size'],
            'gross_pnl': round(row['gross_pnl'], 4),
            'commission': round(row['commission'], 4),
            'net_pnl': round(row['gross_pnl'] - row['commission'], 4),
            'margin_used': round(row['margin_used'], 4),
            'open_at_end': row['open_at_end'],
            'forced': row['forced'],
        })

    def reconciliation_drift(self, final_equity: float, initial_cash: float) -> float:
        booked = sum(t['net_pnl'] for t in self.trades)
        return booked - (final_equity - initial_cash)
