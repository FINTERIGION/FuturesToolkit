"""Strategies package: base class + public examples.

Put private research modules in this folder (gitignored). Import them
directly in ``runner.py``'s strategy registry, e.g.::

    from strategies.your_strategy import YourStrategy
"""

from .base import BarContext, SetupContext, Strategy
from .double_ma import DoubleMaStrategy
from .my_strategy import MyStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy

__all__ = [
    'Strategy', 'SetupContext', 'BarContext',
    'DoubleMaStrategy', 'RsiMeanReversionStrategy', 'MyStrategy',
]
