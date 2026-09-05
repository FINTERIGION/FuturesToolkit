"""Trailing return momentum: information diffuses slowly and positioning
adjusts gradually, so recent winners tend to keep winning for a while (see
docs/factors.md, "时序 + 横截面动量"). ``direction = 1``: a higher trailing
return scores higher.
"""

from __future__ import annotations

from core.params import Int

from .base import Factor
from .primitives import ts_return


class MomentumFactor(Factor):
    """Trailing return over ``lookback`` bars, skipping the most recent
    ``skip`` bars -- the usual short-term-reversal guard, since the last few
    bars of a move are often given back rather than extended.

    Identical formula to
    ``strategies.cross_sectional_momentum.CrossSectionalMomentumStrategy``'s
    hand-written example (``ts_return`` is lifted from it verbatim), so this
    factor and that strategy always agree on what "momentum" means.
    """

    params = {'lookback': 60, 'skip': 5}
    space = {'lookback': Int(20, 120), 'skip': Int(0, 15)}
    constraints = (lambda p: p['skip'] < p['lookback'],)
    direction = 1

    def compute_symbol(self, ctx, sym):
        return ts_return(ctx.close(sym), self.p['lookback'], self.p['skip'])
