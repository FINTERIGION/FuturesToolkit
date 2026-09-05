"""Realized volatility: the low-volatility anomaly -- calmer products tend to
offer a better risk-adjusted return than the market prices in (see
docs/factors.md, "波动率 / 特质波动率"). ``direction = -1``: a *lower*
realized volatility scores higher.
"""

from __future__ import annotations

import numpy as np

from core.params import Int

from .base import Factor
from .primitives import ts_std


class VolatilityFactor(Factor):
    """Rolling standard deviation of daily log returns over ``window`` bars."""

    params = {'window': 20}
    space = {'window': Int(5, 60)}
    direction = -1

    def compute_symbol(self, ctx, sym):
        close = ctx.close(sym)
        with np.errstate(divide='ignore', invalid='ignore'):
            safe_close = np.where(close > 0, close, np.nan)
            log_ret = np.empty_like(close)
            log_ret[0] = np.nan
            log_ret[1:] = np.diff(np.log(safe_close))
        return ts_std(log_ret, self.p['window'])
