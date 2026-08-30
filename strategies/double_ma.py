"""Dual moving-average crossover example strategy."""

import talib

from .base import Strategy


class DoubleMaStrategy(Strategy):
    """Dual moving-average trend-following strategy (example).

    Fast MA above slow MA -> long; fast MA below slow MA -> short.
    ``set_target`` flips directly from one side to the other in a single
    order (no separate close-then-reopen bar).
    """

    params = {'fast_period': 5, 'slow_period': 20, 'lots': 1}

    def setup(self, ctx):
        for sym in ctx.symbols:
            close = ctx.close(sym)
            ctx.add_indicator('fast', sym, talib.SMA(close, self.p['fast_period']))
            ctx.add_indicator('slow', sym, talib.SMA(close, self.p['slow_period']))

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            if not ctx.can_trade(sym):
                continue
            lots = self.p['lots']
            want_long = ctx.ind('fast', sym) > ctx.ind('slow', sym)
            ctx.set_target(sym, lots if want_long else -lots)
