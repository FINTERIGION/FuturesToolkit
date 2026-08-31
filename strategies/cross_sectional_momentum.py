"""Cross-sectional momentum example strategy (multi-product).

Ranks all tradable products by trailing momentum and goes long the leaders,
short the laggards -- the one bundled example that trades the universe as a
single cross-section instead of looping independent per-symbol signals.
"""

from __future__ import annotations

import logging

import numpy as np
import talib

from datafeed.products import product_costs

from .base import Float, Int, Strategy

logger = logging.getLogger(__name__)


def _momentum(close: np.ndarray, lookback: int, skip: int) -> np.ndarray:
    """``close[t-skip] / close[t-skip-lookback] - 1``, with a leading-NaN warmup."""
    out = np.full_like(close, np.nan, dtype='float64')
    span = lookback + skip
    if lookback <= 0 or len(close) <= span:
        return out
    past = close[:-span] if span else close
    recent = close[lookback:len(close) - skip] if skip else close[lookback:]
    with np.errstate(divide='ignore', invalid='ignore'):
        out[span:] = np.where(past > 0, recent / past - 1.0, np.nan)
    return out


class CrossSectionalMomentumStrategy(Strategy):
    """Rank products by trailing momentum; long the leaders, short the laggards.

    Every ``rebalance_days`` bars, rank all tradable products by momentum over
    ``lookback`` bars (skipping the most recent ``skip`` bars, the usual
    short-term-reversal guard). Long the ``top_k`` strongest, short the
    ``top_k`` weakest, flat everything in between. Positions are held
    unchanged between rebalances.

    Sizing per leg is capped two ways: an ATR-based risk budget (each leg
    risks roughly the same ``equity * risk_budget`` on a one-ATR move) and an
    equal split of ``equity * max_gross_margin`` across the ``2 * top_k`` legs
    opened this rebalance. The margin split matters because ``ctx.available``
    only reflects already-filled positions -- sizing every leg off the same
    snapshot of available margin would let each leg assume it has the whole
    account, over-committing margin by roughly the number of legs.
    """

    params = {
        'lookback': 60,
        'skip': 5,
        'atr_period': 20,
        'top_k': 2,
        'rebalance_days': 5,
        'risk_budget': 0.01,
        'max_gross_margin': 0.6,
        'min_universe': 4,
    }
    space = {
        'lookback': Int(20, 120),
        'skip': Int(0, 15),
        'atr_period': Int(10, 40),
        'top_k': Int(1, 3),
        'rebalance_days': Int(1, 20),
    }
    fixed_params = ('risk_budget', 'max_gross_margin', 'min_universe')
    constraints = (lambda p: p['skip'] < p['lookback'],)

    def setup(self, ctx):
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
            close = ctx.close(sym)
            ctx.add_indicator('mom', sym, _momentum(close, self.p['lookback'], self.p['skip']))
            ctx.add_indicator(
                'atr', sym,
                talib.ATR(ctx.high(sym), ctx.low(sym), close, timeperiod=self.p['atr_period']),
            )

    def on_bar(self, ctx):
        if ctx.i < self._next_rebalance:
            return

        tradable = [s for s in ctx.symbols if ctx.can_trade(s)]
        scores = {s: ctx.ind('mom', s) for s in tradable}
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
