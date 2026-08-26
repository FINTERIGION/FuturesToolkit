"""Custom strategy template. Copy or edit for private research."""

import backtrader.indicators as btind

from .base import FuturesStrategyBase


class MyStrategy(FuturesStrategyBase):
    """
    Custom strategy template.

    Usage:
      1. Define the indicators you need in __init__
      2. Implement trading logic in next()
      3. Or add a new file under strategies/ and import it from main.py
      4. Set STRATEGY = MyStrategy in main.py

    Available methods:
      self.buy_signal()                 open long (default product)
      self.buy_signal(symbol='FG')      open long on a named product
      self.sell_signal(symbol='CF')     open short
      self.close_signal(symbol='SA')    close one product
      self.get_position_size()          default product net lots
      self.get_position_size('FG')      one product's net lots
      self.has_pending(symbol='FG')     pending order on that product
      self.is_session(symbol='FG')      True if that product can fill today

    Available data:
      self.data / self.weighted         default product's OI-weighted series
      self.contracts['SA2505']          that product's real contracts
      self.products['FG'].weighted      any loaded product's weighted series
      self.products['FG'].contracts     that product's real contracts
      self.get_weighted('CF')           helper for the line above
      self.get_contract('FG2505')       any real contract, or None
      self.symbols                      products loaded for this run
      self.data.close[0]                today's default weighted close
      self.data.open / high / low / volume / openinterest / settle

    Multi-product sketch (one independent signal per product)::

        for sym in self.symbols:
            if self.has_pending(sym):
                continue
            pos = self.get_position_size(sym)
            close = self.get_weighted(sym).close[0]
            if pos == 0 and close > self.sma[sym][0]:
                self.buy_signal(symbol=sym)
    """

    params = (
        ('symbol', 'SA'),
        # Add strategy parameters here, e.g.:
        # ('fast', 5),
        # ('slow', 20),
    )

    def __init__(self):
        super().__init__()
        # Define indicators here, e.g.:
        # self.fast_ma = btind.SMA(self.data.close, period=self.p.fast)
        # self.slow_ma = btind.SMA(self.data.close, period=self.p.slow)
        pass

    def next(self):
        # Wait if there is a pending order
        if self._pending_order:
            return

        pos = self.get_position_size()

        # -------------------------------------------------------
        # Write your trading logic here.
        # Example: simple price momentum.
        # -------------------------------------------------------
        # if self.data.close[0] > self.data.close[-1]:  # price rose today
        #     if pos <= 0:
        #         if pos < 0:
        #             self.close_signal()
        #         else:
        #             self.buy_signal()
        # else:
        #     if pos > 0:
        #         self.close_signal()
        pass
