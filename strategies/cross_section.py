"""Shared cross-sectional trading logic: rank a score, hold the extremes, size the legs.

Factored out of ``CrossSectionalMomentumStrategy`` so the bridged factor
strategies in :mod:`strategies.factor_bridge` trade a factor exactly the way
that hand-written example does. The sizing in particular is subtle enough
(two independent caps, one of which depends on how many legs open *this*
rebalance) that a second copy would drift.

A user of this mixin supplies the scores; everything from ranking onward --
selection, sizing, and the dropped-leg accounting -- lives here. It reads its
knobs straight off ``self.p``, so the host strategy must declare all of
:data:`REQUIRED_PARAMS`. Splice :data:`DEFAULT_PARAMS`, :data:`DEFAULT_SPACE`
and :data:`FIXED_PARAMS` into the host's own class attributes rather than
restating the values -- the three describe the *trading rule*, so a host that
copies them drifts away from every other host the moment one is retuned.
"""

from __future__ import annotations

import logging

import numpy as np
import talib

from core.params import Int
from datafeed.products import product_costs

logger = logging.getLogger(__name__)

REQUIRED_PARAMS = (
    'top_k', 'rebalance_days', 'atr_period', 'risk_budget', 'max_gross_margin', 'min_universe',
)

DEFAULT_PARAMS = {
    'top_k': 2,
    'rebalance_days': 5,
    'atr_period': 20,
    'risk_budget': 0.01,
    'max_gross_margin': 0.6,
    'min_universe': 4,
}

DEFAULT_SPACE = {
    'top_k': Int(1, 3),
    'rebalance_days': Int(1, 20),
    'atr_period': Int(10, 40),
}

FIXED_PARAMS = ('risk_budget', 'max_gross_margin', 'min_universe')


class CrossSectionMixin:
    """Mix into a ``Strategy`` alongside its own ``setup``/``on_bar``.

    Call :meth:`setup_cross_section` from the host's ``setup`` and
    :meth:`rebalance` from its ``on_bar``, passing whatever ``score_of``
    callable produces the ranking. ``on_finish`` below is picked up
    automatically through the MRO as long as the host does not define its own.
    """

    def setup_cross_section(self, ctx) -> None:
        """Initialize rebalance state and register the ATR series sizing needs.

        Call from the host's ``setup``, before or after registering its own
        indicators -- order does not matter, ``add_indicator`` accumulates the
        warmup requirement across all of them.
        """
        missing = [name for name in REQUIRED_PARAMS if name not in self.p]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: missing required param(s) for CrossSectionMixin: {missing}"
            )
        self._next_rebalance = -1
        self._selected_counts: dict = {}
        self._zero_lot_counts: dict = {}
        self._zero_lot_warned: set = set()
        if len(ctx.symbols) < self.p['min_universe']:
            logger.warning(
                "%d product(s) loaded but min_universe=%d: this strategy ranks a "
                "cross-section and will never rebalance. Pass more --symbols, or "
                "lower min_universe.",
                len(ctx.symbols), self.p['min_universe'],
            )
        for sym in ctx.symbols:
            ctx.add_indicator(
                'atr', sym,
                talib.ATR(ctx.high(sym), ctx.low(sym), ctx.close(sym), timeperiod=self.p['atr_period']),
            )

    def rebalance(self, ctx, score_of) -> None:
        """Rank ``score_of(sym)`` across tradable products and set targets.

        Call from the host's ``on_bar``. ``score_of`` is any callable taking a
        symbol and returning a float; non-finite scores drop the product from
        this bar's cross-section rather than sorting as an extreme.
        """
        if ctx.i < self._next_rebalance:
            return

        tradable = [s for s in ctx.symbols if ctx.can_trade(s)]
        scores = {s: score_of(s) for s in tradable}
        scores = {s: v for s, v in scores.items() if np.isfinite(v)}
        if len(scores) < self.p['min_universe']:
            return

        ranked = sorted(scores, key=scores.get, reverse=True)
        k = min(self.p['top_k'], len(ranked) // 2)
        longs = set(ranked[:k])
        shorts = set(ranked[-k:]) if k else set()
        n_legs = len(longs) + len(shorts)

        for sym in tradable:
            if sym in longs:
                ctx.set_target(sym, self._lots(ctx, sym, n_legs))
            elif sym in shorts:
                ctx.set_target(sym, -self._lots(ctx, sym, n_legs))
            else:
                ctx.close(sym)

        self._next_rebalance = ctx.i + self.p['rebalance_days']

    def _lots(self, ctx, sym, n_legs):
        """Lots capped by ATR-based risk and by an equal share of gross margin."""
        if n_legs <= 0:
            return 0
        self._selected_counts[sym] = self._selected_counts.get(sym, 0) + 1
        atr_value = ctx.ind('atr', sym)
        if atr_value <= 0:
            self._note_zero_lots(
                sym, f'ATR={atr_value:.4g} is not positive',
                'Check the ATR series for this product.',
            )
            return 0
        costs = product_costs(sym)
        risk_lots = int((ctx.equity * self.p['risk_budget']) // (atr_value * costs['multiplier']))

        price = ctx.bar(sym).close
        margin_per_lot = price * costs['multiplier'] * costs['margin_rate']
        if margin_per_lot <= 0:
            self._note_zero_lots(
                sym, f'margin per lot is not positive (close={price:.4g})',
                'Check the price series and product costs for this product.',
            )
            return 0
        leg_budget = ctx.equity * self.p['max_gross_margin'] / n_legs
        margin_lots = int(leg_budget // margin_per_lot)

        lots = max(0, min(risk_lots, margin_lots))
        if lots == 0:
            binding = 'risk_budget' if risk_lots <= margin_lots else 'max_gross_margin'
            self._note_zero_lots(
                sym,
                f'ATR={atr_value:.2f} -> risk cap {risk_lots} lots, '
                f'margin cap {margin_lots} lots; {binding} binds',
                f'Raise {binding} or --cash.',
            )
        return lots

    def _note_zero_lots(self, sym, cause, advice):
        """Record -- and, once per symbol, warn about -- a selected leg that
        sized to zero lots, whatever the reason: both caps rounding down, or a
        degenerate ATR/price that short-circuits sizing entirely.

        Every path in ``_lots`` that returns 0 for a selected symbol must come
        through here, or ``on_finish`` reports a dropped/selected ratio whose
        numerator is missing drops the denominator already counted.

        Not an error: the leg just stays flat. But it unbalances the
        cross-section silently -- the ranking still assigns ``top_k`` names to
        a side while fewer than ``top_k`` actually trade -- and a product whose
        ATR is large relative to ``equity * risk_budget`` can sit out an entire
        run without emitting a single line. Deliberately *not* floored to one
        lot: forcing a leg open would spend more risk than ``risk_budget``
        authorized, which is the one number this sizing exists to respect.

        Warned once per symbol (a rebalance every few bars would otherwise emit
        hundreds of identical lines); ``on_finish`` reports the full tally.
        """
        self._zero_lot_counts[sym] = self._zero_lot_counts.get(sym, 0) + 1
        if sym in self._zero_lot_warned:
            return
        self._zero_lot_warned.add(sym)
        logger.warning(
            "%s sized to 0 lots (%s): leg dropped, cross-section left unbalanced. %s",
            sym, cause, advice,
        )

    def on_finish(self, engine) -> None:
        """Tally dropped legs, so a one-off warning does not read like a one-off
        event when a symbol was in fact skipped on every rebalance it won."""
        if not self._zero_lot_counts:
            return
        summary = ', '.join(
            f'{sym} {n}/{self._selected_counts.get(sym, n)}'
            for sym, n in sorted(self._zero_lot_counts.items(), key=lambda kv: -kv[1])
        )
        logger.warning('Legs dropped to 0 lots (dropped/selected): %s', summary)
