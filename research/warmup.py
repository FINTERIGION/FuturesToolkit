"""Precise, strategy-agnostic warmup-length probing.

``Strategy.setup()`` is where indicators get registered via
``ctx.add_indicator``, which is what actually advances ``Engine.warmup_index``
(see ``strategies/base.py``). Running just that -- not the full day loop --
gives an exact, cheap answer for any strategy, so research code never has to
guess or hardcode a pad length the way a strategy-specific tool would.
"""

from __future__ import annotations

from core.engine import Engine
from core.market import MarketData
from strategies.base import SetupContext


def probe_warmup(market: MarketData, strategy_cls: type, params: dict) -> int:
    """Bar index at which ``strategy_cls(**params)`` first has every
    registered indicator valid, on the given ``market``."""
    strategy = strategy_cls(**params)
    engine = Engine(market, strategy)
    strategy.setup(SetupContext(engine))
    return engine.warmup_index
