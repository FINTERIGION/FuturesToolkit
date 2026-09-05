"""Factor definitions: a score per product per bar, judged without a backtest.

A ``Factor`` is deliberately smaller than a ``Strategy`` -- it has no orders,
no sizing, no margin, just a score. :func:`compute_factor` turns one into a
:class:`FactorPanel`, which :mod:`research.factor_eval` and
:mod:`research.factor_corr` judge directly, and :mod:`strategies.factor_bridge`
turns into a runnable, tunable strategy when a factor is worth trading.

``FactorContext`` mirrors ``strategies.base.SetupContext``'s per-symbol
accessors so a factor and a strategy read data the same way; :meth:`panel`
adds the whole-cross-section view most factors actually want.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from core.market import MarketData, tradable_mask

logger = logging.getLogger(__name__)

__all__ = ['Factor', 'FactorContext', 'FactorPanel', 'compute_factor']

_FIELDS = ('open', 'high', 'low', 'close', 'settle', 'volume', 'oi')


class FactorContext:
    """Full-panel (numpy) view handed to ``Factor.compute``/``compute_symbol``."""

    def __init__(self, market: MarketData, symbols: Sequence[str]):
        missing = [s for s in symbols if s not in market.products]
        if missing:
            raise ValueError(f"Unknown symbol(s) not in this market: {missing}")
        self.market = market
        self.symbols = list(symbols)
        self.dates = market.dates

    def _field(self, sym: str, field: str) -> np.ndarray:
        if sym not in self.market.products:
            raise ValueError(f"Unknown symbol {sym!r}")
        return self.market.products[sym].weighted[field]

    def open(self, sym: str) -> np.ndarray:
        return self._field(sym, 'open')

    def high(self, sym: str) -> np.ndarray:
        return self._field(sym, 'high')

    def low(self, sym: str) -> np.ndarray:
        return self._field(sym, 'low')

    def close(self, sym: str) -> np.ndarray:
        return self._field(sym, 'close')

    def settle(self, sym: str) -> np.ndarray:
        return self._field(sym, 'settle')

    def volume(self, sym: str) -> np.ndarray:
        return self._field(sym, 'volume')

    def oi(self, sym: str) -> np.ndarray:
        return self._field(sym, 'oi')

    def contracts(self, sym: str) -> Dict[str, object]:
        """Per-contract lifecycle data for ``sym`` -- ``{code: ContractSeries}``.

        For factors that need more than the OI-weighted continuous series --
        term structure/carry compares real, simultaneously live contracts
        against each other, which the weighted series cannot represent.
        """
        if sym not in self.market.products:
            raise ValueError(f"Unknown symbol {sym!r}")
        return self.market.products[sym].contracts

    def panel(self, field: str) -> np.ndarray:
        """``float64[n_bars, n_symbols]`` of one OHLCV field across ``self.symbols``,
        in that order -- the vectorized alternative to calling the per-symbol
        accessor in a loop."""
        if field not in _FIELDS:
            raise ValueError(f"Unknown field {field!r}; expected one of {_FIELDS}")
        return np.column_stack([self.market.products[s].weighted[field] for s in self.symbols])


@dataclass
class FactorPanel:
    """A factor's score, already oriented by its ``direction``.

    ``values`` is ``float64[n_bars, n_symbols]`` in ``symbols`` order.
    ``tradable`` is the same shape (see ``core.market.tradable_mask``) --
    a factor can and does score a bar a product could not actually have
    traded (warmup ran ahead of listing, e.g.), and every consumer downstream
    must gate on this mask, not on ``np.isfinite(values)`` alone.
    """

    name: str
    symbols: List[str]
    values: np.ndarray
    tradable: np.ndarray
    direction: int = 1

    @property
    def n_bars(self) -> int:
        return self.values.shape[0]

    @property
    def n_symbols(self) -> int:
        return self.values.shape[1]

    def coverage(self) -> np.ndarray:
        """``int[n_bars]``: how many symbols carry both a real quote and a
        finite score on each bar."""
        return (self.tradable & np.isfinite(self.values)).sum(axis=1)


class Factor:
    """Subclass and implement ``compute_symbol`` (one symbol) or override
    ``compute`` directly (the whole cross-section in one vectorized pass).

    ``params``/``space``/``fixed_params``/``constraints`` mirror
    ``strategies.base.Strategy`` exactly, so ``research.space.resolve_space``
    and the Optuna optimizer apply unchanged once a factor is bridged into a
    strategy (see ``strategies.factor_bridge``).

    ``direction`` is +1 when a higher score predicts a higher subsequent
    return, -1 when it predicts a lower one. :func:`compute_factor` bakes it
    into the returned panel's ``values``, so nothing downstream (IC,
    quantile buckets, the bridge) has to know a factor's sign convention.
    """

    params: dict = {}
    space: dict = {}
    fixed_params: tuple = ()
    constraints: tuple = ()
    direction: int = 1

    def __init__(self, **overrides):
        self.p = {**type(self).params, **overrides}

    def compute(self, ctx: "FactorContext") -> Dict[str, np.ndarray]:
        """Score every symbol in ``ctx.symbols``.

        Default stacks :meth:`compute_symbol` over ``ctx.symbols`` in order;
        override this instead when a factor can score the whole
        cross-section faster in one vectorized pass (most cross-sectional
        factors do -- see ``factors.momentum``/``factors.volatility``).
        """
        return {sym: self.compute_symbol(ctx, sym) for sym in ctx.symbols}

    def compute_symbol(self, ctx: "FactorContext", sym: str) -> np.ndarray:
        """Score one symbol. Implement this, or override ``compute`` directly."""
        raise NotImplementedError(
            f"{type(self).__name__} must override `compute` or `compute_symbol`"
        )


def compute_factor(factor: Factor, ctx: FactorContext) -> FactorPanel:
    """Run ``factor`` over ``ctx`` and package the result as a :class:`FactorPanel`.

    Every symbol's array is shape-checked individually against ``(n_bars,)``
    so a bug in one symbol's computation is reported by name rather than as
    an opaque broadcast error. Infinities become NaN (a factor dividing by a
    degenerate price should not silently poison a whole bar's ranking), and
    an all-NaN result is logged rather than raised -- a factor that cannot
    score this particular universe/window is a research finding, not a crash.
    """
    if factor.direction not in (1, -1):
        raise ValueError(
            f"{type(factor).__name__}.direction must be +1 or -1, got {factor.direction!r}"
        )

    n_bars = ctx.market.n_bars
    n_symbols = len(ctx.symbols)
    raw = factor.compute(ctx)

    values = np.full((n_bars, n_symbols), np.nan, dtype='float64')
    for j, sym in enumerate(ctx.symbols):
        if sym not in raw:
            continue
        arr = np.asarray(raw[sym], dtype='float64')
        if arr.shape != (n_bars,):
            raise ValueError(
                f"{type(factor).__name__}.compute[{sym!r}]: expected shape "
                f"({n_bars},), got {arr.shape}"
            )
        values[:, j] = arr

    values = np.where(np.isinf(values), np.nan, values)

    if not np.isfinite(values).any():
        logger.warning(
            "%s produced an all-NaN panel over %d symbol(s); every downstream "
            "statistic will be empty.", type(factor).__name__, n_symbols,
        )

    values = values * factor.direction
    tradable = tradable_mask(ctx.market, ctx.symbols)

    return FactorPanel(
        name=type(factor).__name__,
        symbols=list(ctx.symbols),
        values=values,
        tradable=tradable,
        direction=factor.direction,
    )
