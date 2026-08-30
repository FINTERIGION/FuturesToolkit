"""Custom strategy template. Copy or edit for private research."""

import talib

from .base import Strategy


class MyStrategy(Strategy):
    """Custom strategy template.

    Usage:
      1. Precompute indicators in ``setup`` with ``ctx.add_indicator`` (the
         engine skips ``on_bar`` until every registered indicator has a
         valid value -- no manual NaN checks needed).
      2. Implement trading logic in ``on_bar``.
      3. Or add a new file under strategies/ and register it in runner.py.

    Available on ``BarContext``:
      ctx.bar(sym)                  Bar(open, high, low, close, settle, volume, oi)
      ctx.ind(name, sym)            registered indicator value at this bar
      ctx.position(sym)             net lots (signed)
      ctx.equity / cash / margin_used / available
      ctx.can_trade(sym)            listed + real print today + contract live
      ctx.contract(sym)             today's calendar contract code, e.g. 'SA509'
      ctx.set_target(sym, lots)     target net position; idempotent
      ctx.buy(sym, lots) / sell(sym, lots) / close(sym)
      ctx.set_stop(sym, price=None, distance=None) / cancel_stop(sym)
      ctx.size_for_risk(sym, stop_distance, risk_pct)

    Multi-product sketch (one independent signal per product)::

        for sym in ctx.symbols:
            if not ctx.can_trade(sym):
                continue
            pos = ctx.position(sym)
            if pos == 0 and ctx.bar(sym).close > ctx.ind('sma', sym):
                ctx.set_target(sym, self.p['lots'])
    """

    params = {
        # Add strategy parameters here, e.g.:
        # 'fast': 5,
        # 'slow': 20,
    }

    def setup(self, ctx):
        # Define indicators here, e.g.:
        # for sym in ctx.symbols:
        #     ctx.add_indicator('sma', sym, talib.SMA(ctx.close(sym), self.p['period']))
        pass

    def on_bar(self, ctx):
        # -------------------------------------------------------
        # Write your trading logic here.
        # Example: simple price momentum.
        # -------------------------------------------------------
        # for sym in ctx.symbols:
        #     if not ctx.can_trade(sym):
        #         continue
        #     pos = ctx.position(sym)
        #     bar = ctx.bar(sym)
        #     if bar.close > bar.open and pos <= 0:
        #         ctx.set_target(sym, 1)
        #     elif bar.close < bar.open and pos >= 0:
        #         ctx.set_target(sym, -1)
        pass
