"""Custom strategy template. Copy or edit for private research."""

# Unused here on purpose: this is a template, and the commented
# `ctx.add_indicator` example in `setup` below needs it the moment
# anyone uncomments it. A linter will flag it; leave it alone.
import talib

from .base import Strategy


class MyStrategy(Strategy):
    """Custom strategy template.

    Usage:
      1. Precompute indicators in ``setup`` with ``ctx.add_indicator`` (the
         engine skips ``on_bar`` until every registered indicator has a
         valid value -- no manual NaN checks needed).
      2. Implement trading logic in ``on_bar``.
      3. Or add a new file under strategies/ -- it is picked up automatically
         by ``strategies.discover_strategies()`` (used by ``ft.py`` and the
         web panel), no registration needed.
      4. Optionally declare a ``space`` alongside ``params`` to say what range
         each one is plausible over, e.g. ``space = {'period': Int(5, 60)}``.
         ``ft.py validate`` perturbs one step within it to check the result
         does not depend on the exact value. If omitted, a range is inferred
         from each param's default (see ``ft.py show-space``).

    Available on ``BarContext``:
      ctx.bar(sym)                  Bar(open, high, low, close, settle, volume, oi)
      ctx.ind(name, sym)            registered indicator value at this bar
      ctx.position(sym)             net lots (signed)
      ctx.equity / cash / margin_used / available
      ctx.can_trade(sym)            listed + real print today + contract live
      ctx.contract(sym)             today's calendar contract code, e.g. 'SA509'
      ctx.set_target(sym, lots)     target net position; idempotent
      ctx.buy(sym, lots) / sell(sym, lots) / close(sym)
      ctx.set_stop(sym, price=...) or set_stop(sym, distance=...) / cancel_stop(sym)
      ctx.set_take_profit(sym, price=...) or (sym, distance=...) / cancel_take_profit(sym)
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
