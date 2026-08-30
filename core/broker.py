"""Cash, positions, margin, commission, and forced liquidation.

Equity/margin model (docs/rewrite-plan.md §5): margin is never deducted from
cash -- it is only checked as a capital-usage limit. Positions are tracked at
weighted-average cost; every fill that reduces or flips a position realizes
``(fill_price - avg_entry) × sign(position) × closed_qty × multiplier``
straight into cash, and commission is deducted from cash at fill time. That
makes

    equity = cash + Σ_positions (mark - avg_entry) × size × multiplier

self-consistent for both longs and shorts -- there is no separate margin term
riding on the equity curve the way there was with backtrader's
``getvalue()``/``getoperationcost()`` split (an 11-13% error in opposite
directions for longs vs. shorts).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from typing import Callable, Dict, List, Optional, Tuple

from datafeed.products import product_costs

from .types import Fill, Reason

SettleLookup = Callable[[str, str], Optional[float]]


@dataclass
class Position:
    symbol: str
    contract: str
    size: int          # signed lots
    avg_entry: float
    last_mark: float    # most recent settle price seen (carried forward on dark bars)


class Broker:
    def __init__(self, initial_cash: float):
        self.cash = float(initial_cash)
        self.positions: Dict[Tuple[str, str], Position] = {}
        self.liquidation_count = 0

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def commission(self, symbol: str, lots: int, price: float) -> float:
        costs = product_costs(symbol)
        if costs['commission_mode'] == 'per_lot':
            return lots * costs['commission_per_lot']
        return lots * price * costs['multiplier'] * costs['commission_rate']

    def fill(
        self,
        symbol: str,
        contract: str,
        size: int,
        price: float,
        bar_index: int,
        date: Date,
        reason: Reason = Reason.SIGNAL,
    ) -> Optional[Fill]:
        """Execute a fill of ``size`` signed lots at ``price``; returns the Fill."""
        if not size:
            return None
        lots = abs(size)
        comm = self.commission(symbol, lots, price)
        self.cash -= comm

        realized = 0.0
        key = (symbol, contract)
        pos = self.positions.get(key)
        if pos is None or pos.size == 0:
            self.positions[key] = Position(symbol, contract, size, price, price)
        elif (size > 0) == (pos.size > 0):
            # Same-direction add: blend the cost basis (linear P&L makes a
            # weighted-average entry exact).
            qty_pos, qty_fill = abs(pos.size), abs(size)
            pos.avg_entry = (
                pos.avg_entry * qty_pos + price * qty_fill
            ) / (qty_pos + qty_fill)
            pos.size += size
        else:
            # Reduce or flip: realize the closed portion into cash now.
            closing = min(abs(pos.size), abs(size))
            sign_pos = 1 if pos.size > 0 else -1
            realized = (price - pos.avg_entry) * sign_pos * closing * product_costs(symbol)['multiplier']
            self.cash += realized
            new_size = pos.size + size
            if new_size == 0:
                del self.positions[key]
            elif (new_size > 0) == (pos.size > 0):
                pos.size = new_size   # partial reduce; cost basis unchanged
                pos.last_mark = price
            else:
                pos.size = new_size   # flipped; leftover opens fresh
                pos.avg_entry = price
                pos.last_mark = price

        return Fill(
            bar_index=bar_index, date=date, symbol=symbol, contract=contract,
            size=size, price=price, commission=comm, reason=reason,
            realized_pnl=realized,
        )

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def net_position(self, symbol: str) -> int:
        return sum(p.size for (sym, _), p in self.positions.items() if sym == symbol)

    def mark_to_market(self, settle_lookup: SettleLookup) -> Tuple[float, float, float]:
        """Value the account at today's settle prices. Does not touch cash.

        ``settle_lookup(symbol, contract)`` returns today's settle price, or
        None if that contract had no print today (dark bar) -- in that case
        the position's last known mark carries forward, matching the ffill
        used for mark-to-market elsewhere in the data pipeline.
        """
        equity = self.cash
        margin_used = 0.0
        for (symbol, contract), pos in self.positions.items():
            price = settle_lookup(symbol, contract)
            if price is not None:
                pos.last_mark = price
            mark = pos.last_mark
            mult = product_costs(symbol)['multiplier']
            margin_rate = product_costs(symbol)['margin_rate']
            equity += (mark - pos.avg_entry) * pos.size * mult
            margin_used += abs(pos.size) * mark * mult * margin_rate
        return equity, margin_used, equity - margin_used

    def force_liquidate(self, bar_index: int, date: Date) -> List[Fill]:
        """Flatten every position at its last mark (call right after
        ``mark_to_market`` so ``last_mark`` reflects today's settle). One
        liquidation event."""
        fills = []
        for (symbol, contract), pos in list(self.positions.items()):
            price = pos.last_mark
            f = self.fill(
                symbol, contract, -pos.size, price, bar_index, date,
                reason=Reason.LIQUIDATION,
            )
            if f is not None:
                fills.append(f)
        if fills:
            self.liquidation_count += 1
        return fills
