"""Strategy API: lifecycle base class plus the two context objects.

See docs/rewrite-plan.md §4. Signals are computed from each product's
OI-weighted continuous series (``SetupContext``/``BarContext`` accessors);
fills happen on that day's calendar contract, resolved by the engine only at
fill time -- strategy code never has to think about which physical contract
an order lands on.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from datafeed.products import product_costs

from core.indicators import guard
from core.params import Categorical, Float, Int, Spec
from core.types import Bar

__all__ = [
    'Strategy', 'SetupContext', 'BarContext', 'Int', 'Float', 'Categorical', 'Spec',
]

logger = logging.getLogger(__name__)


class Strategy:
    """Subclass and implement ``setup``/``on_bar``. ``params`` is a class-level
    dict of defaults; instantiate with ``MyStrategy(**overrides)``.

    ``space`` optionally declares the tunable search space for parameter
    optimization: ``{param_name: Int(...) | Float(...) | Categorical(...)}``.
    Params not listed in ``space`` and not in ``fixed_params`` get a
    heuristic space inferred from their default value (see
    ``research.space.resolve_space``). ``fixed_params`` lists params that
    should never be tuned (position sizing, risk knobs, etc.). ``constraints``
    is a tuple of ``callable(params_dict) -> bool``; a trial whose sampled
    params fail any constraint is pruned rather than scored.
    """

    params: dict = {}
    space: dict = {}
    fixed_params: tuple = ('lots',)
    constraints: tuple = ()

    def __init__(self, **overrides):
        self.p = {**type(self).params, **overrides}

    def setup(self, ctx: "SetupContext") -> None:
        """Called once before the day loop. Precompute full-series indicators here."""

    def on_bar(self, ctx: "BarContext") -> None:
        """Called once per trading day, after warmup. Orders placed here fill
        at the *next* OPEN."""

    def on_finish(self, engine) -> None:
        """Optional: called once after the day loop ends."""


def _first_valid_index(arr: np.ndarray) -> int:
    mask = ~np.isnan(arr)
    if not mask.any():
        return len(arr)
    return int(np.argmax(mask))


class SetupContext:
    """Full-series (numpy) view handed to ``Strategy.setup``."""

    def __init__(self, engine):
        self._engine = engine
        self.symbols = list(engine.symbols)
        self.dates = engine.market.dates

    @property
    def market(self):
        """The full ``MarketData`` this run is trading -- the whole panel,
        not a per-symbol slice. Needed to score the whole universe at once,
        the way ``CrossSectionalMomentumStrategy`` ranks it; most strategies
        want the per-symbol accessors below instead."""
        return self._engine.market

    def _weighted(self, sym: str, field: str) -> np.ndarray:
        return self._engine.market.products[sym].weighted[field]

    def open(self, sym: str) -> np.ndarray:
        return guard(self._weighted(sym, 'open'), name=f'{sym}.open')

    def high(self, sym: str) -> np.ndarray:
        return guard(self._weighted(sym, 'high'), name=f'{sym}.high')

    def low(self, sym: str) -> np.ndarray:
        return guard(self._weighted(sym, 'low'), name=f'{sym}.low')

    def close(self, sym: str) -> np.ndarray:
        return guard(self._weighted(sym, 'close'), name=f'{sym}.close')

    def settle(self, sym: str) -> np.ndarray:
        return guard(self._weighted(sym, 'settle'), name=f'{sym}.settle')

    def volume(self, sym: str) -> np.ndarray:
        return guard(self._weighted(sym, 'volume'), name=f'{sym}.volume')

    def oi(self, sym: str) -> np.ndarray:
        return guard(self._weighted(sym, 'oi'), name=f'{sym}.oi')

    def add_indicator(self, name: str, sym: str, array, *, allow_gaps: bool = False) -> None:
        """Register a precomputed full-series indicator.

        Warmup is per product: this pushes out the bar from which ``sym``
        becomes tradable, and leaves every other product alone. A product whose
        history starts late therefore sits out until its own indicators are
        valid, while the rest of the universe trades from their own first
        valid bar.

        ``allow_gaps`` skips ``guard``'s embedded-NaN check. That check exists
        because TA-Lib silently returns an all-NaN array when a NaN sits
        *inside* the input -- a failure mode specific to TA-Lib's C core, not
        a property indicators in general must have. A cross-sectional score is
        exactly the counterexample: it legitimately has a hole on any bar a
        product could not be scored, and a ranking strategy drops non-finite
        scores per bar (see ``CrossSectionalMomentumStrategy.on_bar``) rather
        than assuming a dense series. Series derived from open interest
        or volume, which go flat or absent around a roll, are the same shape.
        """
        if allow_gaps:
            arr = np.asarray(array, dtype='float64')
        else:
            arr = guard(np.asarray(array, dtype='float64'), name=f'indicator {name}/{sym}')
        self._engine.indicators[(name, sym)] = arr
        self._engine.require_warmup(sym, _first_valid_index(arr))


class BarContext:
    """Per-bar view handed to ``Strategy.on_bar``, after INTRABAR, before SETTLE."""

    def __init__(self, engine, i: int, date):
        self._engine = engine
        self.i = i
        self.date = date
        self.symbols = list(engine.symbols)

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    def bar(self, sym: str) -> Bar:
        w = self._engine.market.products[sym].weighted
        i = self.i
        return Bar(
            open=float(w['open'][i]), high=float(w['high'][i]), low=float(w['low'][i]),
            close=float(w['close'][i]), settle=float(w['settle'][i]),
            volume=float(w['volume'][i]), oi=float(w['oi'][i]),
        )

    def ind(self, name: str, sym: str) -> float:
        return float(self._engine.indicators[(name, sym)][self.i])

    def position(self, sym: str) -> int:
        return self._engine.broker.net_position(sym)

    def can_trade(self, sym: str) -> bool:
        """True when ``sym`` has a session today *and* its own warmup is done."""
        if self.i < self._engine.warmup_by_symbol.get(sym, 0):
            return False
        return self._engine.market.products[sym].can_trade(self.i)

    def queued_orders(self) -> dict:
        """``{symbol: lot delta}`` queued so far this bar, as a copy.

        Empty at the top of every ``on_bar``; the engine folds it into
        ``pending`` afterwards. Reading it is what lets one strategy wrap
        another and veto or shrink its orders without the two having to know
        about each other. Pair it with ``set_target``, which is idempotent per
        bar, to rewrite a decision the wrapped strategy just made.
        """
        return dict(self._engine.queued)

    def contract(self, sym: str) -> str:
        return self._engine.market.products[sym].active_contract(self.i)

    def _valuation(self):
        lookup = self._engine.settle_lookup(self.i)
        return self._engine.broker.mark_to_market(lookup)

    @property
    def equity(self) -> float:
        return self._valuation()[0]

    @property
    def cash(self) -> float:
        return self._engine.broker.cash

    @property
    def margin_used(self) -> float:
        return self._valuation()[1]

    @property
    def available(self) -> float:
        return self._valuation()[2]

    # ------------------------------------------------------------------
    # Orders (target-position semantics)
    # ------------------------------------------------------------------

    def set_target(self, sym: str, lots: int) -> None:
        """Queue whatever delta is needed so the position becomes ``lots``
        after the next OPEN. Idempotent: a repeat call with the same target
        this bar is a no-op; a later call this bar overrides an earlier one."""
        net_now = self._engine.broker.net_position(sym)
        self._engine.queued[sym] = int(lots) - net_now

    def close(self, sym: str) -> None:
        self.set_target(sym, 0)

    def buy(self, sym: str, lots: int = 1) -> None:
        self._engine.queued[sym] = self._engine.queued.get(sym, 0) + int(lots)

    def sell(self, sym: str, lots: int = 1) -> None:
        self._engine.queued[sym] = self._engine.queued.get(sym, 0) - int(lots)

    # ------------------------------------------------------------------
    # Protective bracket: stop and take-profit
    # ------------------------------------------------------------------
    #
    # Both legs arm at the *next* OPEN, against the fill that just happened,
    # and are then checked intrabar against the execution contract's high/low.
    # Prefer ``distance`` to ``price``: signals come off the OI-weighted
    # continuous series but fills land on a real contract, and a distance is
    # anchored on the actual fill so the basis between the two cancels.
    #
    # The two form an OCO pair -- whichever trips first flattens the position
    # and cancels the other. A bar that touches both is resolved by
    # ``Engine._bracket_hit``, which prefers the stop.

    def set_stop(self, sym: str, price: Optional[float] = None, distance: Optional[float] = None) -> None:
        if price is not None:
            self._engine.stop_spec[sym] = {'price': float(price)}
        elif distance is not None:
            self._engine.stop_spec[sym] = {'distance': float(distance)}

    def cancel_stop(self, sym: str) -> None:
        self._engine.stop_spec.pop(sym, None)
        self._engine.live_stop.pop(sym, None)

    def set_take_profit(self, sym: str, price: Optional[float] = None, distance: Optional[float] = None) -> None:
        """Target that closes the whole position when touched intrabar.

        Note this is a *touch* fill, at the target price (or at the open when
        the bar gapped past it). That is the right model for a resting limit
        order, and it is not the same rule as letting the bar close through
        the target and leaving at the next open -- see the measurement in
        ``strategies/momentum_barrier.py``, where the second rule is worth
        roughly twice as much because the trades that blow through the target
        are the ones worth keeping. Which rule suits a strategy is an
        empirical question; this one is opt-in.
        """
        if price is not None:
            self._engine.tp_spec[sym] = {'price': float(price)}
        elif distance is not None:
            self._engine.tp_spec[sym] = {'distance': float(distance)}

    def cancel_take_profit(self, sym: str) -> None:
        self._engine.tp_spec.pop(sym, None)
        self._engine.live_tp.pop(sym, None)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def size_for_risk(self, sym: str, stop_distance: float, risk_pct: float) -> int:
        """Lots such that a full stop-out risks ``risk_pct`` of current equity.

        Returns 0 -- silently, as far as trading goes -- whenever the budget
        does not stretch to a single lot. That is the correct sizing answer
        (opening one anyway would spend more risk than ``risk_pct``
        authorized), but it is indistinguishable from "no signal" in the
        results, so each zero is traced at DEBUG. DEBUG rather than WARNING
        because this is called per bar per symbol: run with ``--verbose`` when
        a strategy trades less than expected.
        """
        if stop_distance <= 0:
            logger.debug('%s: stop_distance=%s is not positive; 0 lots', sym, stop_distance)
            return 0
        multiplier = product_costs(sym)['multiplier']
        per_lot_risk = stop_distance * multiplier
        if per_lot_risk <= 0:
            logger.debug('%s: per-lot risk=%s is not positive; 0 lots', sym, per_lot_risk)
            return 0
        risk_amount = self.equity * risk_pct
        lots = int(risk_amount // per_lot_risk)
        if lots == 0:
            logger.debug(
                '%s: risk budget %.2f is under one lot (stop %.4f x multiplier %s '
                '= %.2f per lot); 0 lots',
                sym, risk_amount, stop_distance, multiplier, per_lot_risk,
            )
        return lots
