"""RSI mean-reversion example strategy."""

import backtrader.indicators as btind

from strategies.base import FuturesStrategyBase


class RsiMeanReversionStrategy(FuturesStrategyBase):
    """
    RSI mean-reversion strategy (example).

    Logic:
      - RSI < oversold   -> oversold, go long
      - RSI > overbought -> overbought, go short
      - Close position when RSI returns to the midline (50)
    """

    params = (
        ('symbol', 'SA'),
        ('rsi_period', 14),
        ('oversold', 30),
        ('overbought', 70),
    )

    def __init__(self):
        super().__init__()
        self.rsi = btind.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        if self._pending_order:
            return

        pos = self.get_position_size()
        rsi_val = self.rsi[0]

        if pos == 0:
            if rsi_val < self.p.oversold:
                self.buy_signal()
            elif rsi_val > self.p.overbought:
                self.sell_signal()
        elif pos > 0 and rsi_val > 50:
            self.close_signal()
        elif pos < 0 and rsi_val < 50:
            self.close_signal()
