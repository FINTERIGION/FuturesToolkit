"""Strategy API: lifecycle base class plus the two context objects.

See docs/rewrite-plan.md §4. Signals are computed from each product's
OI-weighted continuous series (``SetupContext``/``BarContext`` accessors);
fills happen on that day's calendar contract, resolved by the engine only at
fill time -- strategy code never has to think about which physical contract
an order lands on.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from datafeed.products import product_costs

from core.indicators import guard
from core.types import Bar


class Strategy:
    """Subclass and implement ``setup``/``on_bar``. ``params`` is a class-level
    dict of defaults; instantiate with ``MyStrategy(**overrides)``."""

    params: dict = {}

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

    def add_indicator(self, name: str, sym: str, array) -> None:
        """Register a precomputed full-series indicator. The engine skips
        ``on_bar`` until every registered indicator has a valid value."""
        arr = guard(np.asarray(array, dtype='float64'), name=f'indicator {name}/{sym}')
        self._engine.indicators[(name, sym)] = arr
        first_valid = _first_valid_index(arr)
        self._engine.warmup_index = max(self._engine.warmup_index, first_valid)


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
        return self._engine.market.products[sym].can_trade(self.i)

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
    # Protective stop
    # ------------------------------------------------------------------

    def set_stop(self, sym: str, price: Optional[float] = None, distance: Optional[float] = None) -> None:
        if price is not None:
            self._engine.stop_spec[sym] = {'price': float(price)}
        elif distance is not None:
            self._engine.stop_spec[sym] = {'distance': float(distance)}

    def cancel_stop(self, sym: str) -> None:
        self._engine.stop_spec.pop(sym, None)
        self._engine.live_stop.pop(sym, None)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def size_for_risk(self, sym: str, stop_distance: float, risk_pct: float) -> int:
        """Lots such that a full stop-out risks ``risk_pct`` of current equity."""
        if stop_distance <= 0:
            return 0
        multiplier = product_costs(sym)['multiplier']
        per_lot_risk = stop_distance * multiplier
        if per_lot_risk <= 0:
            return 0
        risk_amount = self.equity * risk_pct
        return int(risk_amount // per_lot_risk)
