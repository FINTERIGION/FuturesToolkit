"""Four-phase day loop: OPEN -> INTRABAR -> SIGNAL -> SETTLE.

See docs/rewrite-plan.md §2. Order routing is contract-agnostic at the point
a strategy calls ``ctx.set_target`` / ``ctx.buy`` / ``ctx.sell`` -- the
engine only resolves *which* calendar contract an order lands on at the
moment it actually fills (``MarketData.contract_by_bar``), so there is no
need to detect and re-route a "signal placed the day before a roll" the way
the old backtrader strategy base had to.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .broker import Broker
from .ledger import Ledger
from .market import MarketData
from .types import Reason


def _to_date(np_date) -> Date:
    return pd.Timestamp(np_date).date()


class Engine:
    def __init__(
        self,
        market: MarketData,
        strategy,
        initial_cash: float = 100_000.0,
        slippage: float = 0.0,
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
        self.warmup_index = 0

        self.equity_records: List[dict] = []
        self.signal_log: List[dict] = []
        self._prev_equity = initial_cash

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
            if delta:
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
            self.ledger.finish(self.broker, last_date)

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
        }
