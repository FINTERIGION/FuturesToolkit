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

from datafeed.products import tick_size

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
        # In ticks, not price points: a fill is charged
        # ``slippage × tick_size(symbol)``, so one setting means the same thing
        # on gold (0.02 per tick) as on copper (10 per tick).
        self.slippage = float(slippage)

        self.broker = Broker(initial_cash)
        self.ledger = Ledger()

        # Orders the account could not fund, and the bar the account died on.
        # Both are engine-only facts: a strategy sees neither its own rejected
        # order nor the run ending early, so they are reported at the end.
        self.rejections: List[dict] = []
        # Rolls that had to leave a contract at its carried mark because the
        # leg being rolled out of had already printed for the last time. Those
        # fills happened at a price nobody traded, so they are disclosed rather
        # than buried -- see ``_maybe_roll``.
        self.stranded_rolls: List[dict] = []
        # Orders dropped because the product's own indicators were not valid
        # yet, aggregated per product. Invisible to the strategy that placed
        # them, so they are counted here and reported at the end.
        self.warmup_skips: Dict[str, dict] = {}
        self.blown_up = False
        self.blown_up_date: Optional[Date] = None
        self.blown_up_bar: Optional[int] = None

        self.pending: Dict[str, int] = {}      # queued from last SIGNAL phase, fills this OPEN
        self.deferred: Dict[str, int] = {}      # accumulated while dark, replays once live
        self.queued: Dict[str, int] = {}        # this bar's SIGNAL-phase order deltas
        # Two protective legs of one bracket. The ``*_spec`` half is the rule
        # and is sticky -- it survives a dark bar and re-arms after a close and
        # reopen -- while the ``live_*`` half is the order actually resting on
        # a contract right now. A resting order carries the ``anchor`` it was
        # resolved from (see ``_bracket_anchor``), and ``_arm_brackets``
        # re-resolves it whenever that anchor moves, so the level always
        # reflects the current rule and the current position rather than
        # whichever of the two happened to come first. Either leg filling
        # cancels both (OCO), in ``_intrabar_phase``.
        self.stop_spec: Dict[str, dict] = {}    # {'distance': x} or {'price': x}, sticky
        self.live_stop: Dict[str, dict] = {}    # {'price','contract','anchor'} currently resting
        self.tp_spec: Dict[str, dict] = {}      # take-profit, same shape as stop_spec
        self.live_tp: Dict[str, dict] = {}

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

    def _held_legs(self, sym: str) -> List[tuple]:
        """``[(contract, size)]`` for every contract ``sym`` is positioned in.

        The engine maintains a one-leg-per-product invariant (``_maybe_roll``
        moves the whole position onto the calendar contract, and
        ``_fill_signal`` only ever adds to the leg already held), so this
        returns at most one pair. It returns a list anyway because that
        invariant is worth being able to *check* rather than assume: brackets,
        the ledger and the margin model are all keyed by product, and a second
        leg appearing behind their backs is the kind of thing that shows up as
        a mispriced stop or a reconciliation drift three layers away.
        """
        return [
            (contract, pos.size)
            for (s, contract), pos in self.broker.positions.items()
            if s == sym and pos.size != 0
        ]

    def _current_contract(self, sym: str) -> Optional[str]:
        """The single contract ``sym`` is positioned in, or None when flat."""
        legs = self._held_legs(sym)
        if not legs:
            return None
        if len(legs) > 1:
            # Never expected. Reported rather than silently resolved, because
            # everything downstream (one bracket per product, one ledger row
            # per product) would otherwise describe only part of the book.
            logger.error(
                '%s is split across %d contracts (%s) -- the one-leg invariant '
                'is broken; brackets and the trade log will describe only the '
                'largest leg. The next roll folds them back together.',
                sym, len(legs), ', '.join(f'{c}:{s:+d}' for c, s in legs),
            )
        return max(legs, key=lambda leg: abs(leg[1]))[0]

    def _has_future_print(self, sym: str, contract: str, i: int) -> bool:
        """Whether ``contract`` prints again at or after bar ``i``.

        False means the contract is finished as far as this run can see, so
        waiting for a real price on it is waiting for something that will
        never arrive. On a *sliced* market (a research window) this answers
        the question locally -- a contract still alive past ``window.end`` is
        finished within the window -- which is the right answer for a backtest
        that ends there anyway.
        """
        panel = self.market.products.get(sym)
        series = panel.contracts.get(contract) if panel is not None else None
        if series is None or not len(series.live_bars):
            return False
        return int(series.live_bars[-1]) >= i

    def _slipped(self, open_price: float, size: int, sym: str) -> float:
        """Fill price after slippage, which is charged in ticks of ``sym``."""
        if not self.slippage:
            return open_price
        offset = self.slippage * tick_size(sym)
        if size > 0:
            return open_price + offset
        return open_price - offset

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
        """Hold this bar's order over: the product had no session to fill it on.

        The protective bracket is left exactly as it is. A dark bar cannot
        fill either leg anyway -- ``_intrabar_phase`` has no row to check them
        against -- so clearing them here, as this used to, bought nothing and
        cost a good deal: ``_arm_brackets`` put them straight back at the end
        of the same phase, and a ``distance`` leg came back re-resolved
        against whatever the position's cost had since become. Which meant a
        non-trading day could move a stop, and two runs differing only in
        where the holidays fell did not agree.
        """
        amt = self.pending.pop(sym, 0)
        if amt:
            self.deferred[sym] = self.deferred.get(sym, 0) + amt

    def _maybe_roll(self, sym: str, i: int, date: Date) -> None:
        """Move ``sym``'s whole position onto today's calendar contract.

        Each held leg is rolled **by its own size**, never by the product's
        net across contracts: closing ``net_position`` on one leg overshoots
        the moment a product is split across two, which flips that leg short
        and grows the book every bar it repeats.

        A leg whose contract did not print today has no exit price, and what
        happens then depends on whether one is still coming:

        * **Still trading** -- an ordinary dark day. The roll waits, and
          ``_fill_signal`` keeps new orders on this same leg meanwhile, so
          the wait cannot itself split the product.
        * **Finished printing** -- expired, or delisted mid-run. Waiting is
          waiting forever, so the leg is closed at its carried mark and the
          exposure moves on. That fill is at a price nobody traded, so it is
          recorded in ``stranded_rolls`` and warned about at the end of the
          run rather than passed off as an ordinary roll.
        """
        panel = self.market.products[sym]
        target = panel.active_contract(i)
        if not target:
            return
        legs = [(c, size) for c, size in self._held_legs(sym) if c != target]
        if not legs:
            return
        new_row = self._row_for(sym, target, i)
        if new_row is None:
            return  # target has no print today; delay the roll
        new_open = float(new_row[0])

        rolled = []
        for contract, size in legs:
            old_row = self._row_for(sym, contract, i)
            if old_row is not None:
                old_ref = float(old_row[0])
                stranded = False
            elif not self._has_future_print(sym, contract, i):
                old_ref = self.broker.positions[(sym, contract)].last_mark
                stranded = True
            else:
                continue  # alive, just dark today -- its roll waits for a print

            close_size = -size
            close_price = self._slipped(old_ref, close_size, sym)
            f = self.broker.fill(sym, contract, close_size, close_price, i, date,
                                 reason=Reason.ROLL)
            if f:
                self.ledger.process_fill(f)

            open_price = self._slipped(new_open, size, sym)
            f = self.broker.fill(sym, target, size, open_price, i, date, reason=Reason.ROLL)
            if f:
                self.ledger.process_fill(f)

            if stranded:
                self.stranded_rolls.append({
                    'date': date, 'bar_index': i, 'symbol': sym,
                    'from_contract': contract, 'to_contract': target,
                    'size': size, 'mark': old_ref,
                })
                logger.warning(
                    '%s: %s printed for the last time before bar %d, so its %+d lot(s) '
                    'were rolled into %s at the carried mark %.4f -- a price that was '
                    'never traded.', date, contract, i, size, target, old_ref,
                )
            rolled.append((abs(size), old_ref))

        if rolled:
            # Basis of the leg that actually carries the bracket (the largest,
            # which is what `_current_contract` reports). Unslipped on both
            # sides so the shift is the market's basis, not the fill's.
            _, ref = max(rolled, key=lambda pair: pair[0])
            self._reanchor_brackets(sym, new_open - ref)

    def _reanchor_brackets(self, sym: str, basis: float) -> None:
        """Put both protective legs back on the price scale they now trade on.

        A roll changes which contract the bracket is checked against, and two
        calendar contracts run a basis of a few percent. Migrating only the
        *contract* and leaving the *level* -- which is what this used to do --
        leaves a level derived from the old contract's prices being compared
        against the new one's high/low: on a backwardated roll that puts a
        long's stop above the market, and ``_bracket_hit``'s gap tier flattens
        the position at the very next open for no reason at all.

        A ``distance`` spec needs no arithmetic here: dropping the resting
        order makes ``_arm_brackets`` re-resolve it against the position's new
        ``avg_entry`` -- the price actually paid on the new contract -- so the
        basis cancels by construction. That is the whole reason strategies are
        pointed at distances rather than prices.

        An explicit ``price`` has no such anchor: it is a level the strategy
        named while the old contract was the market. It is shifted by the
        basis the roll just realized, which keeps the level the same distance
        from the market as the strategy chose. The sticky spec is shifted too,
        not just the resting order, so a position that later closes and
        reopens does not re-arm on the stale scale.
        """
        for spec_by_sym, live in ((self.stop_spec, self.live_stop),
                                  (self.tp_spec, self.live_tp)):
            spec = spec_by_sym.get(sym)
            if spec is not None and 'price' in spec:
                spec['price'] = float(spec['price']) + basis
            live.pop(sym, None)   # re-armed against the new leg in `_arm_brackets`

    def _fill_signal(self, sym: str, i: int, date: Date, size: int) -> bool:
        """Fill ``size`` at this bar's open; False if there was no price to fill on.

        Orders land on the leg the product is **already** positioned in, and
        only on the calendar contract when it is flat. By the time this runs
        ``_maybe_roll`` has already moved the position onto the calendar
        contract unless the leg it holds is dark today -- and in that case
        there is no price to fill against, so the order defers exactly as it
        would on any other dark bar. Routing to the calendar contract
        regardless is what used to open a second leg alongside the one waiting
        to roll, splitting the product.
        """
        if not size:
            return True
        contract = self._current_contract(sym) or self.market.products[sym].active_contract(i)
        row = self._row_for(sym, contract, i)
        if row is None:
            return False  # calendar contract has no print today; nothing to fill against
        price = self._slipped(float(row[0]), size, sym)
        if not self.broker.can_afford(sym, contract, size, price):
            self._reject(sym, contract, size, price, i, date)
            return True
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

    def _reject(self, sym: str, contract: str, size: int, price: float,
                i: int, date: Date) -> None:
        """Record an order the account cannot fund.

        Rejected, not deferred: a deferred order is one the *market* could not
        fill and which the same decision still stands behind, so it replays.
        This one the account refused, and replaying it would open the position
        on a later bar at a price no signal chose. A strategy that still wants
        it will ask again next bar, off a fresh signal -- which is also what a
        real rejection leaves you doing.
        """
        equity, margin_used, available = self.broker.valuation()
        self.rejections.append({
            'date': date, 'bar_index': i, 'symbol': sym, 'contract': contract,
            'size': size, 'price': price, 'equity': equity,
            'margin_used': margin_used, 'available': available,
            'margin_required': self.broker.margin_for(sym, size, price),
        })
        logger.debug(
            'Rejected %+d %s @ %.2f: needs %.2f margin, only %.2f available',
            size, contract, price, self.broker.margin_for(sym, size, price), available,
        )

    def _flush_and_fill_pending(self, sym: str, i: int, date: Date) -> None:
        amt = self.deferred.pop(sym, 0) + self.pending.pop(sym, 0)
        if not self._fill_signal(sym, i, date, amt):
            self.deferred[sym] = amt   # keep it queued, same as a dark bar

    def _bracket_anchor(self, sym: str, net: int, contract: str, spec: dict) -> tuple:
        """Everything the resting level was resolved from.

        A live order whose anchor still matches is still the right level; one
        whose anchor moved has to be resolved again. Deciding that on the
        *inputs* rather than on "is an order currently resting" is the whole
        point: the old rule re-armed only when ``live_*`` happened to be empty,
        which made a level depend on unrelated history -- a dark bar cleared
        the resting order and silently re-anchored it, a reversal did not
        clear it and left a short holding the long's stop, and a strategy
        calling ``set_stop`` again while one rested changed nothing at all.
        """
        if 'price' in spec:
            return (contract, float(spec['price']))
        pos = self.broker.positions.get((sym, contract))
        avg_entry = pos.avg_entry if pos is not None else None
        return (contract, 1 if net > 0 else -1, avg_entry, float(spec['distance']))

    def _arm_one(self, sym: str, net: int, contract: str, spec: dict,
                 live: Dict[str, dict], sign: int, anchor: tuple) -> None:
        """Resolve one protective leg into a resting order on ``contract``.

        ``sign`` is -1 for the stop and +1 for the take-profit: a long's stop
        sits ``distance`` below its cost and its target ``distance`` above,
        and a short mirrors both.

        A ``distance`` is resolved against ``Position.avg_entry`` -- the price
        actually paid on the real contract -- rather than against anything
        from the weighted continuous series, because the two run a basis of a
        few percent and a level lifted from the weighted series would land in
        the wrong place. That is why strategies are encouraged to pass
        distances; an explicit ``price`` is taken at face value.
        """
        if 'price' in spec:
            price = spec['price']
        else:
            avg_entry = self.broker.positions[(sym, contract)].avg_entry
            offset = sign * spec['distance']
            price = avg_entry + offset if net > 0 else avg_entry - offset
        live[sym] = {'price': float(price), 'contract': contract, 'anchor': anchor}

    def _arm_brackets(self, i: int, date: Date) -> None:
        """Arm stop and take-profit at the end of OPEN, against today's fill.

        Each leg is re-resolved whenever what it was resolved *from* has moved
        -- the position's cost, its side, or the contract it sits on -- and
        left alone otherwise. ``*_spec`` is the source of truth; a spec that
        has gone away takes its resting order with it.
        """
        for sym in self.symbols:
            net = self.broker.net_position(sym)
            if net == 0:
                # stale orders from a position that's since closed
                self.live_stop.pop(sym, None)
                self.live_tp.pop(sym, None)
                continue
            contract = self._current_contract(sym)
            if contract is None:
                continue
            for spec_by_sym, live, sign in (
                (self.stop_spec, self.live_stop, -1),
                (self.tp_spec, self.live_tp, +1),
            ):
                spec = spec_by_sym.get(sym)
                if not spec:
                    live.pop(sym, None)
                    continue
                anchor = self._bracket_anchor(sym, net, contract, spec)
                resting = live.get(sym)
                if resting is not None and resting.get('anchor') == anchor:
                    continue
                self._arm_one(sym, net, contract, spec, live, sign, anchor)

    def _open_phase(self, i: int, date: Date) -> None:
        for sym in self.symbols:
            panel = self.market.products[sym]
            if not panel.can_trade(i):
                self._defer_symbol(sym)
                continue
            self._maybe_roll(sym, i, date)
            self._flush_and_fill_pending(sym, i, date)
        self._arm_brackets(i, date)

    # ------------------------------------------------------------------
    # INTRABAR phase
    # ------------------------------------------------------------------

    @staticmethod
    def _bracket_hit(net: int, o: float, h: float, l: float,
                     stop: Optional[float], target: Optional[float]):
        """``(fill_price, Reason)`` for whichever leg trips today, else None.

        Two tiers, checked in order, because a daily bar records no path:

        1. **Gapped through at the open.** Whichever level the open is already
           past fills *at the open* -- the bar never traded better than that.
           This tier has to come first: a long that gaps above its target never
           had the chance to trade back down to its stop, so ranking the stop
           ahead of it would book a loss the day the position won.
        2. **Touched during the session.** Here the open sits between the two
           levels and high/low say only that a level was reached, never when.
           The **stop wins** -- the pessimistic read, so a bracket can never
           flatter a backtest by assuming the target came first.

        With ``target`` None this is exactly the old stop-only rule.
        """
        if net > 0:
            if stop is not None and o <= stop:
                return o, Reason.STOP
            if target is not None and o >= target:
                return o, Reason.TAKE_PROFIT
            if stop is not None and l <= stop:
                return stop, Reason.STOP
            if target is not None and h >= target:
                return target, Reason.TAKE_PROFIT
        elif net < 0:
            if stop is not None and o >= stop:
                return o, Reason.STOP
            if target is not None and o <= target:
                return o, Reason.TAKE_PROFIT
            if stop is not None and h >= stop:
                return stop, Reason.STOP
            if target is not None and l <= target:
                return target, Reason.TAKE_PROFIT
        return None

    def _intrabar_phase(self, i: int, date: Date) -> None:
        for sym in sorted(set(self.live_stop) | set(self.live_tp)):
            stop_info = self.live_stop.get(sym)
            tp_info = self.live_tp.get(sym)
            # Both legs are armed off `_current_contract` and both migrate on a
            # roll, so they never disagree; either one names the contract.
            contract = (stop_info or tp_info)['contract']
            row = self._row_for(sym, contract, i)
            if row is None:
                continue
            o, h, l = float(row[0]), float(row[1]), float(row[2])
            # The position on the contract the bracket is actually resting on,
            # not the product's net across contracts: filling `-net` on one leg
            # of a split product overshoots it into the opposite direction.
            pos = self.broker.positions.get((sym, contract))
            net = pos.size if pos is not None else 0
            if net == 0:
                self.live_stop.pop(sym, None)
                self.live_tp.pop(sym, None)
                continue
            hit = self._bracket_hit(
                net, o, h, l,
                stop_info['price'] if stop_info else None,
                tp_info['price'] if tp_info else None,
            )
            if hit is None:
                continue
            fill_price, reason = hit
            fsize = -net
            f = self.broker.fill(sym, contract, fsize, fill_price, i, date, reason=reason)
            if f:
                self.ledger.process_fill(f)
                self.signal_log.append({
                    'date': date, 'price': fill_price,
                    'direction': 'buy' if fsize > 0 else 'sell',
                    'size': fsize, 'comm': f.commission, 'symbol': sym,
                })
            # OCO: one leg filling flattens the position, so the other is void.
            self.live_stop.pop(sym, None)
            self.stop_spec.pop(sym, None)
            self.live_tp.pop(sym, None)
            self.tp_spec.pop(sym, None)

    # ------------------------------------------------------------------
    # SIGNAL phase
    # ------------------------------------------------------------------

    def _signal_phase(self, i: int, date: Date, bar_context_cls) -> None:
        self.queued = {}
        if i >= self.warmup_index:
            ctx = bar_context_cls(self, i, date)
            self.strategy.on_bar(ctx)
        for sym, delta in self.queued.items():
            if not delta:
                continue
            # ``ctx.can_trade`` already reports a still-warming product as
            # untradable, but a strategy is free not to ask; dropping the order
            # here makes per-product warmup hold whatever the strategy does.
            # Dropped, and *counted*: this is the one order the engine refuses
            # that the strategy has no way of noticing, and until it was
            # reported it looked from the outside exactly like a signal that
            # never fired -- see ``_report_unfinished``.
            if i < self.warmup_by_symbol.get(sym, 0):
                self._note_warmup_skip(sym, i, delta)
                continue
            self.pending[sym] = self.pending.get(sym, 0) + delta

    def _note_warmup_skip(self, sym: str, i: int, delta: int) -> None:
        """Book one order dropped because ``sym``'s own warmup is not done.

        Aggregated per product rather than kept row by row: a strategy with no
        ``can_trade`` guard signals on every bar of its warmup for every
        product, so a list would grow with the universe and tell the reader
        nothing a count and a bar range do not.
        """
        row = self.warmup_skips.get(sym)
        if row is None:
            self.warmup_skips[sym] = {
                'symbol': sym, 'n_orders': 1, 'lots': abs(delta),
                'first_bar': i, 'last_bar': i, 'ready_at': self.warmup_by_symbol.get(sym, 0),
            }
            return
        row['n_orders'] += 1
        row['lots'] += abs(delta)
        row['last_bar'] = i

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

        # Insolvent even after the forced liquidation had its chance. Every bar
        # after this one would be fiction -- positions opened on capital that
        # does not exist -- so the run stops here and says so.
        if equity <= 0 and not self.blown_up:
            self.blown_up = True
            self.blown_up_date = date
            self.blown_up_bar = i

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
        last_bar = n - 1
        for i in range(n):
            date = _to_date(self.market.dates[i])
            self._open_phase(i, date)
            self._intrabar_phase(i, date)
            self._signal_phase(i, date, bar_context_cls)
            self._settle_phase(i, date)
            if self.blown_up:
                last_bar = i
                break

        if self.market.n_bars:
            last_date = _to_date(self.market.dates[last_bar])
            self.ledger.finish(self.broker, last_date, last_bar=last_bar)

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
            'rejections': self.rejections,
            'stranded_rolls': self.stranded_rolls,
            'warmup_skips': dict(self.warmup_skips),
            'blown_up': self.blown_up,
            'blown_up_date': self.blown_up_date,
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

        if self.stranded_rolls:
            by_symbol: Dict[str, int] = {}
            for r in self.stranded_rolls:
                by_symbol[r['symbol']] = by_symbol.get(r['symbol'], 0) + 1
            logger.warning(
                "%d roll(s) had to exit a contract at its carried mark because it had "
                "stopped printing (%s). Those exits are at prices that were never "
                "traded, so the P&L across them is an estimate. Usually this means the "
                "product's `main_months` in datafeed/products.py no longer match where "
                "the liquidity is, and the calendar is holding contracts to the end of "
                "their life -- check them against the cached open interest.",
                len(self.stranded_rolls),
                ', '.join(f'{sym} x{n}' for sym, n in sorted(by_symbol.items())),
            )

        if self.warmup_skips:
            worst = max(self.warmup_skips.values(), key=lambda r: r['n_orders'])
            logger.warning(
                "%d order(s) across %d product(s) were dropped because the product's "
                "own indicators were not valid yet (%s). The strategy signalled them "
                "and never learned they went nowhere -- guard each symbol with "
                "`ctx.can_trade(sym)` to see this coming. Worst: %s, %d order(s) over "
                "bars %d-%d, tradable from bar %d.",
                sum(r['n_orders'] for r in self.warmup_skips.values()),
                len(self.warmup_skips),
                ', '.join(f"{r['symbol']} x{r['n_orders']}"
                          for r in sorted(self.warmup_skips.values(), key=lambda r: r['symbol'])),
                worst['symbol'], worst['n_orders'], worst['first_bar'],
                worst['last_bar'], worst['ready_at'],
            )

        if self.rejections:
            by_symbol: Dict[str, int] = {}
            for r in self.rejections:
                by_symbol[r['symbol']] = by_symbol.get(r['symbol'], 0) + 1
            worst = max(self.rejections, key=lambda r: r['margin_required'] - r['available'])
            logger.warning(
                "%d order(s) rejected for insufficient margin (%s). The account "
                "could not fund them, so the strategy traded a smaller book than "
                "it asked for -- the metrics below describe what was affordable, "
                "not what was signalled. Worst shortfall: %s on %s needed %s "
                "margin against %s available. Raise --cash or trade fewer "
                "products to run the strategy as written.",
                len(self.rejections),
                ', '.join(f'{sym} x{n}' for sym, n in sorted(by_symbol.items())),
                worst['contract'], worst['date'],
                f"{worst['margin_required']:,.0f}", f"{worst['available']:,.0f}",
            )

        if self.blown_up:
            traded = (self.blown_up_bar or 0) + 1
            final = (self.equity_records[-1]['equity'] if self.equity_records
                     else self.broker.valuation()[0])
            logger.warning(
                "BLOWN UP on %s (bar %d of %d): equity reached %s even after the "
                "forced liquidation, so the run stopped there and the remaining "
                "%d bar(s) were never traded. Every metric below covers only the "
                "run up to the failure.",
                self.blown_up_date, traded, n_bars, f'{final:,.2f}', n_bars - traded,
            )
