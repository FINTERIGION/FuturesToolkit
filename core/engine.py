"""Four-phase day loop: OPEN -> INTRABAR -> SIGNAL -> SETTLE.

See docs/rewrite-plan.md §2. Order routing is contract-agnostic at the point
a strategy calls ``ctx.set_target`` / ``ctx.buy`` / ``ctx.sell`` -- the
engine only resolves *which* calendar contract an order lands on at the
moment it actually fills (``MarketData.contract_by_bar``), so there is no
need to detect and re-route a "signal placed the day before a roll" the way
the old backtrader strategy base had to.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .broker import Broker
from .ledger import Ledger
from .market import MarketData
from .types import Reason

logger = logging.getLogger(__name__)


def _to_date(np_date) -> Date:
    return pd.Timestamp(np_date).date()


class Engine:
    def __init__(
        self,
        market: MarketData,
        strategy,
        initial_cash: float = 100_000.0,
        slippage: float = 0.0,
        warmup_bars: int = 0,
    ):
        self.market = market
        self.symbols = market.symbols
        self.strategy = strategy
        self.slippage = float(slippage)

        self.broker = Broker(initial_cash)
        self.ledger = Ledger()

        self.pending: Dict[str, int] = {}      # queued from last SIGNAL phase, fills this OPEN
        self.deferred: Dict[str, int] = {}      # accumulated while dark, replays once live
        self.queued: Dict[str, int] = {}        # this bar's SIGNAL-phase order deltas
        self.stop_spec: Dict[str, dict] = {}    # {'distance': x} or {'price': x}, sticky
        self.live_stop: Dict[str, dict] = {}    # {'price', 'contract'} currently resting

        self.indicators: Dict[tuple, "object"] = {}
        # Warmup is tracked per product, not once for the whole universe: a
        # product listed late (or simply missing early history) must not hold
        # the rest of the universe out of the market. ``require_warmup`` only
        # ever raises a symbol's entry, never lowers it, so the caller-supplied
        # floor -- used by research/runner_api.py to hide a leading pad window
        # from both trading and the equity curve -- survives registration.
        self._warmup_floor = int(warmup_bars)
        self.warmup_by_symbol: Dict[str, int] = {
            sym: self._warmup_floor for sym in self.symbols
        }
        # Floor on ``record_start``, kept separate from ``warmup_index`` only
        # so that indicator registration can never make the equity curve start
        # *earlier* than the caller asked for.
        self._record_from = int(warmup_bars)

        self.equity_records: List[dict] = []
        self.signal_log: List[dict] = []
        self._prev_equity = initial_cash

    @property
    def warmup_index(self) -> int:
        """First bar on which *any* product can trade.

        This gates the signal phase as a whole: there is no point calling
        ``on_bar`` before the earliest-warming product is ready, but waiting
        for the slowest one would throw away every other product's history.
        Per-product readiness is enforced separately, in ``warmup_by_symbol``.
        """
        if not self.warmup_by_symbol:
            return self._warmup_floor
        return min(self.warmup_by_symbol.values())

    @property
    def warmup_full(self) -> int:
        """First bar on which *every* product can trade.

        This is the pad a caller needs if it wants the whole universe live from
        the first bar of a window -- see ``research.warmup.probe_warmup``.
        """
        if not self.warmup_by_symbol:
            return self._warmup_floor
        return max(self.warmup_by_symbol.values())

    def require_warmup(self, sym: str, first_valid: int) -> int:
        """Hold ``sym`` out of the market until bar ``first_valid``.

        Monotonic on purpose -- it raises a product's warmup and never lowers
        it -- so indicators registered in any order settle on the strictest
        one, and the caller-supplied floor is never undercut. A symbol the
        market does not carry starts from that same floor rather than from
        zero. This is the only supported way to move ``warmup_by_symbol``.
        """
        current = self.warmup_by_symbol.get(sym, self._warmup_floor)
        bar = max(current, int(first_valid))
        self.warmup_by_symbol[sym] = bar
        return bar

    @property
    def record_start(self) -> int:
        """First bar that appears in ``equity_records``.

        Never earlier than ``warmup_index``, which indicator registration
        raises past the caller's floor whenever the registered indicators need
        more history than the supplied ``warmup_bars`` covers. Bars in
        ``[warmup_bars, warmup_index)`` are ones ``_signal_phase`` skips
        entirely, so recording them would prepend a run of flat, zero-return
        bars that no decision of the strategy's produced -- deflating the
        window's Sharpe, volatility and capital exposure by an amount that
        varies with each parameter set's lookback, i.e. unevenly across the
        trials of one study. ``research/runner_api.run_window`` reports the
        absolute bar this lands on as ``effective_start``.
        """
        return max(self._record_from, self.warmup_index)

    # ------------------------------------------------------------------
    # Contract / price helpers
    # ------------------------------------------------------------------

    def _current_contract(self, sym: str) -> Optional[str]:
        for (s, c), pos in self.broker.positions.items():
            if s == sym and pos.size != 0:
                return c
        return None

    def _slipped(self, open_price: float, size: int) -> float:
        if size > 0:
            return open_price + self.slippage
        return open_price - self.slippage

    def _row_for(self, sym: str, contract: str, i: int) -> Optional[np.ndarray]:
        """``contract``'s OHLCV row at bar ``i``, or None if it did not print.

        Also returns None for a contract the panel has no series for at all,
        so callers never have to trust that ``contract_by_bar`` and
        ``ProductPanel.contracts`` agree.
        """
        panel = self.market.products.get(sym)
        if panel is None or not contract:
            return None
        series = panel.contracts.get(contract)
        if series is None:
            return None
        return series.row_at(i)

    def settle_lookup(self, i: int) -> Callable[[str, str], Optional[float]]:
        def lookup(symbol: str, contract: str) -> Optional[float]:
            row = self._row_for(symbol, contract, i)
            if row is None:
                return None
            return float(row[4])  # open high low close settle oi volume

        return lookup

    # ------------------------------------------------------------------
    # OPEN phase
    # ------------------------------------------------------------------

    def _defer_symbol(self, sym: str) -> None:
        amt = self.pending.pop(sym, 0)
        if amt:
            self.deferred[sym] = self.deferred.get(sym, 0) + amt
        self.live_stop.pop(sym, None)   # spec stays; resting order does not survive a dark bar

    def _maybe_roll(self, sym: str, i: int, date: Date) -> None:
        panel = self.market.products[sym]
        target = panel.active_contract(i)
        current = self._current_contract(sym)
        if not current or current == target:
            return
        net = self.broker.net_position(sym)
        if net == 0:
            return
        old_row = self._row_for(sym, current, i)
        if old_row is None:
            return  # old contract didn't print today; delay the roll
        new_row = self._row_for(sym, target, i)
        if new_row is None:
            return  # target has no print today either; delay the roll

        close_size = -net
        close_price = self._slipped(float(old_row[0]), close_size)
        f = self.broker.fill(sym, current, close_size, close_price, i, date, reason=Reason.ROLL)
        if f:
            self.ledger.process_fill(f)

        open_size = net
        open_price = self._slipped(float(new_row[0]), open_size)
        f = self.broker.fill(sym, target, open_size, open_price, i, date, reason=Reason.ROLL)
        if f:
            self.ledger.process_fill(f)

        if sym in self.live_stop:
            self.live_stop[sym]['contract'] = target

    def _fill_signal(self, sym: str, i: int, date: Date, size: int) -> bool:
        """Fill ``size`` at this bar's open; False if there was no price to fill on."""
        if not size:
            return True
        contract = self.market.products[sym].active_contract(i)
        row = self._row_for(sym, contract, i)
        if row is None:
            return False  # calendar contract has no print today; nothing to fill against
        price = self._slipped(float(row[0]), size)
        f = self.broker.fill(sym, contract, size, price, i, date, reason=Reason.SIGNAL)
        if f is None:
            return True
        self.ledger.process_fill(f)
        self.signal_log.append({
            'date': date, 'price': price,
            'direction': 'buy' if size > 0 else 'sell',
            'size': size, 'comm': f.commission, 'symbol': sym,
        })
        return True

    def _flush_and_fill_pending(self, sym: str, i: int, date: Date) -> None:
        amt = self.deferred.pop(sym, 0) + self.pending.pop(sym, 0)
        if not self._fill_signal(sym, i, date, amt):
            self.deferred[sym] = amt   # keep it queued, same as a dark bar

    def _arm_stops(self, i: int, date: Date) -> None:
        for sym in self.symbols:
            net = self.broker.net_position(sym)
            if net == 0:
                self.live_stop.pop(sym, None)   # stale order from a position that's since closed
                continue
            spec = self.stop_spec.get(sym)
            if not spec or sym in self.live_stop:
                continue
            contract = self._current_contract(sym)
            if contract is None:
                continue
            if 'price' in spec:
                stop_price = spec['price']
            else:
                avg_entry = self.broker.positions[(sym, contract)].avg_entry
                stop_price = avg_entry - spec['distance'] if net > 0 else avg_entry + spec['distance']
            self.live_stop[sym] = {'price': float(stop_price), 'contract': contract}

    def _open_phase(self, i: int, date: Date) -> None:
        for sym in self.symbols:
            panel = self.market.products[sym]
            if not panel.can_trade(i):
                self._defer_symbol(sym)
                continue
            self._maybe_roll(sym, i, date)
            self._flush_and_fill_pending(sym, i, date)
        self._arm_stops(i, date)

    # ------------------------------------------------------------------
    # INTRABAR phase
    # ------------------------------------------------------------------

    def _intrabar_phase(self, i: int, date: Date) -> None:
        for sym in list(self.live_stop.keys()):
            info = self.live_stop[sym]
            contract = info['contract']
            row = self._row_for(sym, contract, i)
            if row is None:
                continue
            o, h, l = float(row[0]), float(row[1]), float(row[2])
            stop = info['price']
            net = self.broker.net_position(sym)
            if net > 0:
                hit = o <= stop or l <= stop
                fill_price = o if o <= stop else stop
            elif net < 0:
                hit = o >= stop or h >= stop
                fill_price = o if o >= stop else stop
            else:
                hit = False
            if not hit:
                continue
            fsize = -net
            f = self.broker.fill(sym, contract, fsize, fill_price, i, date, reason=Reason.STOP)
            if f:
                self.ledger.process_fill(f)
                self.signal_log.append({
                    'date': date, 'price': fill_price,
                    'direction': 'buy' if fsize > 0 else 'sell',
                    'size': fsize, 'comm': f.commission, 'symbol': sym,
                })
            self.live_stop.pop(sym, None)
            self.stop_spec.pop(sym, None)

    # ------------------------------------------------------------------
    # SIGNAL phase
    # ------------------------------------------------------------------

    def _signal_phase(self, i: int, date: Date, bar_context_cls) -> None:
        self.queued = {}
        if i >= self.warmup_index:
            ctx = bar_context_cls(self, i, date)
            self.strategy.on_bar(ctx)
        for sym, delta in self.queued.items():
            # ``ctx.can_trade`` already reports a still-warming product as
            # untradable, but a strategy is free not to ask; dropping the order
            # here makes per-product warmup hold whatever the strategy does.
            if delta and i >= self.warmup_by_symbol.get(sym, 0):
                self.pending[sym] = self.pending.get(sym, 0) + delta

    # ------------------------------------------------------------------
    # SETTLE phase
    # ------------------------------------------------------------------

    def _settle_phase(self, i: int, date: Date) -> None:
        lookup = self.settle_lookup(i)
        equity, margin_used, available = self.broker.mark_to_market(lookup)
        if available < 0 and self.broker.positions:
            liq_fills = self.broker.force_liquidate(i, date)
            for f in liq_fills:
                self.ledger.process_fill(f)
            equity, margin_used, available = self.broker.mark_to_market(lookup)

        if i < self.record_start:
            # Warmup/pad window: valued for bookkeeping continuity only.
            # Excluded from the equity curve so a research window's metrics
            # reflect only bars the strategy actually traded on, not the
            # history before it -- neither the caller's pad nor the extra
            # bars an indicator's own lookback pushed the start out by.
            self._prev_equity = equity
            return

        prev = self._prev_equity
        daily_return = (equity - prev) / prev if prev else 0.0
        position = {sym: self.broker.net_position(sym) for sym in self.symbols}
        self.equity_records.append({
            'date': date, 'equity': equity, 'position': position,
            'daily_return': daily_return, 'margin_used': margin_used, 'available': available,
        })
        self._prev_equity = equity

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_backtest(self, setup_context_cls, bar_context_cls) -> dict:
        setup_ctx = setup_context_cls(self)
        self.strategy.setup(setup_ctx)

        n = self.market.n_bars
        for i in range(n):
            date = _to_date(self.market.dates[i])
            self._open_phase(i, date)
            self._intrabar_phase(i, date)
            self._signal_phase(i, date, bar_context_cls)
            self._settle_phase(i, date)

        if self.market.n_bars:
            last_date = _to_date(self.market.dates[-1])
            self.ledger.finish(self.broker, last_date, last_bar=self.market.n_bars - 1)

        self._report_unfinished()

        on_finish = getattr(self.strategy, 'on_finish', None)
        if callable(on_finish):
            on_finish(self)

        return {
            'equity_records': self.equity_records,
            'trade_logs': self.ledger.trades,
            'signal_log': self.signal_log,
            'broker': self.broker,
            'ledger': self.ledger,
            'liquidation_count': self.broker.liquidation_count,
            'deferred': {sym: amt for sym, amt in self.deferred.items() if amt},
        }

    def _report_unfinished(self) -> None:
        """Warn about the two ways a run can end with nothing to show for
        itself that neither the metrics nor the trade log can explain.

        Both are facts only the engine holds -- a strategy cannot see either
        one -- and both otherwise surface as a silent zero-trade result, which
        reads like "the signal never fired" when the truth is "the run never
        gave it the chance to". ``compute_metrics`` returns ``{}`` for the
        first case, so without this the CLI prints a table of zeros whose
        ``Initial Cash: 0.00`` sends the reader after ``--cash`` instead of
        after the lookback that actually caused it.
        """
        n_bars = self.market.n_bars
        if self.record_start >= n_bars:
            logger.warning(
                "No bar was traded or recorded: indicators need %d warmup bars "
                "but the market has only %d. Every metric will be empty. "
                "Shorten the strategy's lookback, or widen the date range.",
                self.record_start, n_bars,
            )

        leftover = {sym: amt for sym, amt in self.deferred.items() if amt}
        if leftover:
            logger.warning(
                "Run ended with order(s) still deferred and never filled: %s. "
                "Each product had no tradable session after its signal fired -- "
                "these lots are absent from both the trade log and the equity curve.",
                leftover,
            )
