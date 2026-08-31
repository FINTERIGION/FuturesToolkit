"""RSI mean-reversion example strategy."""

import talib

from .base import Int, Strategy


class RsiMeanReversionStrategy(Strategy):
    """RSI mean-reversion strategy (example).

    RSI < oversold -> go long; RSI > overbought -> go short; close when RSI
    crosses back through the midline (50).
    """

    params = {'rsi_period': 14, 'oversold': 30, 'overbought': 70, 'lots': 1}
    space = {
        'rsi_period': Int(5, 40),
        'oversold': Int(10, 40),
        'overbought': Int(60, 90),
    }
    constraints = (lambda p: p['oversold'] < p['overbought'],)

    def setup(self, ctx):
        for sym in ctx.symbols:
            ctx.add_indicator('rsi', sym, talib.RSI(ctx.close(sym), self.p['rsi_period']))

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            if not ctx.can_trade(sym):
                continue
            pos = ctx.position(sym)
            rsi_val = ctx.ind('rsi', sym)
            lots = self.p['lots']

            if pos == 0:
                if rsi_val < self.p['oversold']:
                    ctx.set_target(sym, lots)
                elif rsi_val > self.p['overbought']:
                    ctx.set_target(sym, -lots)
            elif pos > 0 and rsi_val > 50:
                ctx.close(sym)
            elif pos < 0 and rsi_val < 50:
                ctx.close(sym)
