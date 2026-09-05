"""Turn any factor into a backtestable, tunable strategy.

``factor_runner.py`` answers "does this score predict anything". This module
answers the next question -- "does trading it make money once sizing, margin,
commission and rolls are in the way" -- without asking the user to write a
strategy at all.

One strategy class is generated per factor ``factors.discover_factors()``
finds, and published into this module's namespace so ``discover_strategies()``
picks it up like any hand-written one. Write ``factors/carry.py`` and you
immediately get::

    python runner.py --strategy factor_carry --param lookback=90
    python research_runner.py optimize --strategy factor_carry

with the factor's own params and the trading params tuned jointly, because
``research.space.resolve_space`` reads the merged ``params``/``space`` off the
generated class and neither knows nor cares that it was generated.

The trading rule is the same cross-section every bundled example uses (see
:class:`strategies.cross_section.CrossSectionMixin`): long the ``top_k``
highest-scoring products, short the ``top_k`` lowest, rebalanced every
``rebalance_days`` bars. It is deliberately plain -- the point is to isolate
the factor's contribution, so a bridged backtest is a baseline to beat with a
purpose-built strategy, not a finished one.
"""

from __future__ import annotations

import logging

from factors import discover_factors
from factors.base import FactorContext, compute_factor

from . import base
from .cross_section import CrossSectionMixin, DEFAULT_PARAMS, DEFAULT_SPACE, FIXED_PARAMS

logger = logging.getLogger(__name__)

__all__ = ['make_factor_strategy', 'FACTOR_STRATEGIES']

_SCORE = '_factor_score'


def _factor_setup(self, ctx) -> None:
    fctx = FactorContext(ctx.market, ctx.symbols)
    factor = self.factor_cls(**{k: v for k, v in self.p.items() if k in self.factor_cls.params})
    panel = compute_factor(factor, fctx)
    for i, sym in enumerate(panel.symbols):
        ctx.add_indicator(_SCORE, sym, panel.values[:, i])
    self.setup_cross_section(ctx)


def _factor_on_bar(self, ctx) -> None:
    self.rebalance(ctx, lambda sym: ctx.ind(_SCORE, sym))


def make_factor_strategy(factor_cls: type, name: str) -> type:
    """Build a ``Strategy`` subclass that trades ``factor_cls`` cross-sectionally.

    The generated class merges the factor's ``params``/``space``/
    ``fixed_params``/``constraints`` with the trading ones. A name declared by
    both is an error rather than a silent overwrite: whichever won, one half of
    the strategy would be reading a value meant for the other.
    """
    factor_params = dict(getattr(factor_cls, 'params', {}) or {})
    factor_space = dict(getattr(factor_cls, 'space', {}) or {})
    factor_fixed = set(getattr(factor_cls, 'fixed_params', ()) or ())
    factor_constraints = tuple(getattr(factor_cls, 'constraints', ()) or ())

    clash = set(factor_params) & set(DEFAULT_PARAMS)
    if clash:
        raise ValueError(
            f"factor {name!r}: param(s) {sorted(clash)} clash between the factor and "
            f"the cross-section trading rule; rename one side."
        )

    params = {**factor_params, **DEFAULT_PARAMS}
    space = {**factor_space, **DEFAULT_SPACE}
    fixed_params = tuple(sorted(factor_fixed | set(FIXED_PARAMS)))

    class_name = f'Factor{name.title().replace("_", "")}Strategy'
    return type(
        class_name,
        (CrossSectionMixin, base.Strategy),
        {
            '__module__': __name__,
            '__doc__': f"Cross-sectional strategy generated from the {name!r} factor.",
            'factor_cls': factor_cls,
            'setup': _factor_setup,
            'on_bar': _factor_on_bar,
            'params': params,
            'space': space,
            'fixed_params': fixed_params,
            'constraints': factor_constraints,
        },
    )


def _generate_all() -> dict:
    """One strategy per discovered factor, published into this module.

    A factor that cannot be bridged (a param name clashing with the trading
    rule's) is logged and skipped rather than raised: this runs at import time,
    on the path of ``discover_strategies()``, and one bad private factor should
    not take down ``runner.py --help`` for every other strategy.
    """
    generated = {}
    for name, factor_cls in discover_factors().items():
        try:
            generated[f'factor_{name}'] = make_factor_strategy(factor_cls, name)
        except ValueError as e:
            logger.warning("Skipping factor bridge for %r: %s", name, e)
    return generated


FACTOR_STRATEGIES = _generate_all()
globals().update(FACTOR_STRATEGIES)
