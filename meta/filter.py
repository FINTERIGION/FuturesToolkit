"""The gate: wrap any ``Strategy`` so a meta-model can veto its *entries*.

``make_meta_filtered(SomeStrategy, model)`` returns a ``Strategy`` subclass
that behaves exactly like ``SomeStrategy`` except that orders which would
*increase* exposure must first clear the meta-model. It is strategy-agnostic
-- it reads intents off ``ctx.queued_orders()`` and rewrites them with
``ctx.set_target`` -- so nothing in ``strategies/`` needs to know this exists.

**The one asymmetry that makes meta-labeling sound: exits are never blocked.**
A model that could suppress a close would not be filtering trades, it would be
inventing a new exit rule, and the labels it was trained on (realized P&L
under the primary's *own* exits) would no longer describe the trades it
produces. Only the opening portion of any order is ever gated; the closing
portion of a reversal passes through untouched.

**Why the unfiltered baseline also goes through this wrapper.** Registering
features via ``ctx.add_indicator`` raises ``Engine.warmup_by_symbol`` (see
``strategies/base.py``), which pushes out the first tradable bar. If the
baseline skipped the wrapper it would start earlier than the filtered arm and
the two equity curves would not be comparable. Running the baseline as
``make_meta_filtered(cls, PassThroughModel())`` makes the warmup identical by
construction, so any difference between the arms is the filtering and nothing
else.

**A known second-order effect, not worked around on purpose.** A vetoed entry
is invisible to the wrapped strategy, which has already updated its own state
as though it had entered. Three cases, in increasing order of how much it
matters:

* A stateless strategy -- one that re-derives its target from indicators every
  bar, like ``strategies/double_ma.py`` -- is unaffected entirely. There is no
  state to be wrong.
* A strategy that reconciles against ``ctx.position`` each bar (the convention
  in this repo: clear the entry record when ``pos == 0``, then re-evaluate)
  self-heals on the very next bar.
* For that one bar, such a strategy may believe it holds risk it does not,
  which can marginally shrink a *different* product's size under a
  portfolio-heat cap.

Reaching into the wrapped strategy to fix the third case would make the
wrapper strategy-specific, which costs more than a one-bar effect is worth.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from strategies.base import Strategy

from meta.features import FEATURE_PREFIX, build_feature_arrays, feature_row

logger = logging.getLogger(__name__)

__all__ = ['make_meta_filtered', 'split_order']


def split_order(net: int, target: int) -> Tuple[int, int]:
    """Decompose ``net -> target`` into ``(floor, opening)``.

    ``floor`` is the position the order reaches using only risk-reducing
    moves, and is always allowed. ``opening`` is the extra exposure beyond it,
    and is what the model gets a vote on -- so the vetoed outcome is ``floor``
    and the approved outcome is ``target``.

        +2 -> +3   (2, 1)    adding to a long: the add is gated
        +2 -> +1   (1, 0)    trimming: nothing to gate
        +2 ->  0   (0, 0)    closing: nothing to gate
        +2 -> -2   (0, -2)   reversal: the close happens, the short is gated
         0 -> -1   (0, -1)   fresh short
    """
    if target == 0 or (net != 0 and (net > 0) == (target > 0) and abs(target) <= abs(net)):
        return target, 0                      # purely reducing (or flat): untouched
    if net != 0 and (net > 0) != (target > 0):
        return 0, target                      # reversal: close is free, new side is gated
    if net != 0 and abs(target) > abs(net):
        return net, target - net              # same side, larger: only the add is gated
    return 0, target                          # from flat


def make_meta_filtered(base_cls: type, model) -> type:
    """A ``Strategy`` subclass wrapping ``base_cls``, gated by ``model``.

    The result is an ordinary strategy class: it takes the same ``params``,
    and works unchanged with ``run_single_backtest``, ``run_window`` and
    ``probe_warmup``.
    """

    class _MetaFiltered(Strategy):
        params = dict(getattr(base_cls, 'params', {}))
        space = dict(getattr(base_cls, 'space', {}))
        fixed_params = tuple(getattr(base_cls, 'fixed_params', ('lots',)))
        constraints = tuple(getattr(base_cls, 'constraints', ()))

        wrapped_cls = base_cls
        meta_model = model

        def __init__(self, **overrides):
            super().__init__(**overrides)
            self.base = base_cls(**overrides)
            self.arrays: Dict[Tuple[str, str], np.ndarray] = {}
            # Every judgement made on the most recent bar, keyed by symbol.
            # `meta_runner.py signal` reads this to explain today's decision;
            # it is per-bar, not cumulative, so it cannot grow without bound.
            self.last_decisions: Dict[str, dict] = {}
            self.n_vetoed = 0
            self.n_allowed = 0
            self.n_uncovered = 0

        def setup(self, ctx) -> None:
            self.base.setup(ctx)
            self.arrays = build_feature_arrays(ctx)
            for (name, sym), arr in self.arrays.items():
                # Registered for its warmup side effect: features must be
                # valid before the gate can judge anything, and this is the
                # supported way to say so. The gate reads `self.arrays`
                # directly, since it needs values at an arbitrary bar.
                ctx.add_indicator(f'{FEATURE_PREFIX}{name}', sym, arr)

        def on_bar(self, ctx) -> None:
            self.last_decisions = {}
            self.base.on_bar(ctx)

            for sym, delta in ctx.queued_orders().items():
                if not delta:
                    continue
                net = ctx.position(sym)
                target = net + delta
                floor, opening = split_order(net, target)
                if opening == 0:
                    continue

                side = 1 if opening > 0 else -1
                verdict = self._judge(ctx, sym, side)
                self.last_decisions[sym] = {
                    'net': net, 'target': target, 'floor': floor, 'side': side, **verdict,
                }
                if not verdict['allowed']:
                    ctx.set_target(sym, floor)

        def on_finish(self, engine) -> None:
            on_finish = getattr(self.base, 'on_finish', None)
            if callable(on_finish):
                on_finish(engine)
            total = self.n_allowed + self.n_vetoed
            if total or self.n_uncovered:
                # Attempts, not trades. A vetoed signal is not consumed: the
                # product stays flat, so the primary re-proposes it on every
                # bar its condition still holds. That makes this ratio lower
                # than the model's `keep_rate`, which was calibrated on
                # realized trades -- the two are different denominators, not a
                # miscalibration.
                logger.info(
                    'Meta gate: %d/%d entry attempts allowed (%.1f%%), %d vetoed, '
                    '%d passed through uncovered.',
                    self.n_allowed, total, 100.0 * self.n_allowed / total if total else 0.0,
                    self.n_vetoed, self.n_uncovered,
                )

        # --------------------------------------------------------------

        def _judge(self, ctx, sym: str, side: int) -> dict:
            # Features are indexed by the *local* bar, because `self.arrays`
            # was built from whatever market this run was handed. The model is
            # addressed by date, which means the same thing whether this run
            # is the full history or a sliced research window -- see
            # `WalkForwardMetaModel`.
            x = feature_row(self.arrays, sym, ctx.i, side)
            if x is None:
                self.n_uncovered += 1
                logger.debug('%s bar %d: features incomplete; entry passed through', sym, ctx.i)
                return {'allowed': True, 'proba': None, 'threshold': None, 'reason': 'no_features'}

            p = self.meta_model.proba(sym, ctx.date, x)
            if p is None:
                self.n_uncovered += 1
                return {'allowed': True, 'proba': None, 'threshold': None, 'reason': 'no_model'}

            thr = self.meta_model.threshold(ctx.date)
            allowed = p >= thr
            self.n_allowed += int(allowed)
            self.n_vetoed += int(not allowed)
            return {
                'allowed': bool(allowed), 'proba': float(p), 'threshold': float(thr),
                'reason': 'allowed' if allowed else 'vetoed',
            }

    _MetaFiltered.__name__ = f'MetaFiltered{base_cls.__name__}'
    _MetaFiltered.__qualname__ = _MetaFiltered.__name__
    _MetaFiltered.__doc__ = (
        f'{base_cls.__name__} with entries gated by a '
        f'{type(model).__name__} meta-model.'
    )
    return _MetaFiltered


def unfiltered(base_cls: type) -> type:
    """``base_cls`` through the wrapper with nothing gated -- the fair baseline.

    Identical warmup and identical feature computation to the filtered arm, so
    a difference between the two is the model's doing. See the module
    docstring.
    """
    from meta.model import PassThroughModel

    return make_meta_filtered(base_cls, PassThroughModel())
