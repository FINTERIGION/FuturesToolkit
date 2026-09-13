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
    # Most recent price seen for this contract: today's settle once SETTLE has
    # run, and the fill price when one lands before that. Every fill updates
    # it, adds included -- ``valuation()`` is what the OPEN-phase margin check
    # reads, and it runs mid-bar, after some symbols have filled and before
    # today's settle exists. Leaving an add on yesterday's mark, as this used
    # to, sized the margin of every symbol behind it in the loop off a stale
    # price. Carried forward unchanged on a dark bar.
    last_mark: float


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
            pos.last_mark = price
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

    def valuation(self) -> Tuple[float, float, float]:
        """``(equity, margin_used, available)`` at each position's carried mark.

        The same arithmetic as ``mark_to_market`` with no new prices to apply,
        so there is one valuation formula here rather than two that can drift
        apart. Safe to call mid-bar: it neither needs today's settle -- which
        does not exist yet at OPEN, where the margin check runs -- nor moves
        any position's mark.
        """
        return self.mark_to_market(lambda symbol, contract: None)

    def margin_for(self, symbol: str, size: int, price: float) -> float:
        """Initial margin on ``size`` lots of ``symbol`` marked at ``price``."""
        costs = product_costs(symbol)
        return abs(size) * price * costs['multiplier'] * costs['margin_rate']

    def can_afford(self, symbol: str, contract: str, size: int, price: float) -> bool:
        """Whether filling ``size`` lots at ``price`` leaves the account solvent.

        An order that does not *raise* the margin requirement always passes --
        closes, reductions, and flips into a cheaper leg. That exemption is the
        point: a hard rejection must never trap a position inside a margin
        call, which is exactly when the account most needs to get out. Anything
        that raises the requirement has to leave ``equity >= margin_used`` once
        filled, or it is refused.

        Only strategy orders are put through this. A roll is the same exposure
        on a different contract and a forced liquidation is the remedy for
        insolvency, so neither is something the account can decline.
        """
        equity, margin_used, _ = self.valuation()
        pos = self.positions.get((symbol, contract))
        held = pos.size if pos is not None else 0
        before = self.margin_for(symbol, held, pos.last_mark) if pos is not None else 0.0
        after = margin_used - before + self.margin_for(symbol, held + size, price)
        if after <= margin_used:
            return True
        return equity - after >= 0

    def force_liquidate(
        self, bar_index: int, date: Date,
        price_for: Optional[Callable[[str, int, float], float]] = None,
    ) -> List[Fill]:
        """Flatten every position at its last mark (call right after
        ``mark_to_market`` so ``last_mark`` reflects today's settle). One
        liquidation event.

        ``price_for(symbol, size, mark)`` turns each mark into the fill price.
        A forced liquidation is a market order like any other, so the engine
        passes its slippage here; the broker knows nothing about ticks. Left
        out, the fill lands exactly on the mark.
        """
        fills = []
        for (symbol, contract), pos in list(self.positions.items()):
            size = -pos.size
            price = pos.last_mark if price_for is None else price_for(symbol, size, pos.last_mark)
            f = self.fill(
                symbol, contract, size, price, bar_index, date,
                reason=Reason.LIQUIDATION,
            )
            if f is not None:
                fills.append(f)
        if fills:
            self.liquidation_count += 1
        return fills
