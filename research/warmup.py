"""Precise, strategy-agnostic warmup-length probing.

``Strategy.setup()`` is where indicators get registered via
``ctx.add_indicator``, which is what actually fills in
``Engine.warmup_by_symbol`` (see ``strategies/base.py``). Running just that --
not the full day loop -- gives an exact, cheap answer for any strategy, so
research code never has to guess or hardcode a pad length the way a
strategy-specific tool would.
"""

from __future__ import annotations

from core.engine import Engine
from core.market import MarketData
from strategies.base import SetupContext


def probe_warmup(market: MarketData, strategy_cls: type, params: dict) -> int:
    """Bar index at which ``strategy_cls(**params)`` first has every
    registered indicator valid, on the given ``market``.

    Warmup is per product, so this reports the slowest one (``warmup_full``):
    the pad has to cover the whole universe for a window to open with every
    product live. Products still cold at the window's start simply sit out
    until their own warmup completes -- they no longer hold the others back.
    """
    strategy = strategy_cls(**params)
    engine = Engine(market, strategy)
    strategy.setup(SetupContext(engine))
    return engine.warmup_full
