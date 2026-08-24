"""Dual moving-average crossover example strategy."""

import backtrader.indicators as btind

from strategies.base import FuturesStrategyBase


class DoubleMaStrategy(FuturesStrategyBase):
    """
    Dual moving-average trend-following strategy (example).

    Logic:
      - Fast MA above slow MA -> long
      - Fast MA below slow MA -> short
      - On a reverse, close first; the next bar still sees the same MA
        state and opens the new position (fills the following open)
    """

    params = (
        ('symbol', 'SA'),
        ('fast_period', 5),
        ('slow_period', 20),
    )

    def __init__(self):
        super().__init__()
        self.fast_ma = btind.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = btind.SMA(self.data.close, period=self.p.slow_period)

    def next(self):
        # Wait if there is a pending order
        if self._pending_order:
            return

        pos = self.get_position_size()
        # Use MA *state*, not CrossOver pulse: after a close the next bar
        # still wants the opposite side and will open it.
        want_long = self.fast_ma[0] > self.slow_ma[0]

        if want_long:
            if pos < 0:
                self.close_signal()
            elif pos == 0:
                self.buy_signal()
        else:
            if pos > 0:
                self.close_signal()
            elif pos == 0:
                self.sell_signal()
