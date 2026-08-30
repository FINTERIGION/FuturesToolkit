"""Shared data structures passed between engine, broker, and ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from enum import Enum
from typing import NamedTuple, Optional


class OrderType(Enum):
    MARKET = 'market'
    STOP = 'stop'


class Reason(Enum):
    """Why a fill happened. Rolls and liquidations don't count as logical
    trade entries/exits in the ledger; signal and stop fills do."""
    SIGNAL = 'signal'
    ROLL = 'roll'
    STOP = 'stop'
    LIQUIDATION = 'liquidation'


class Bar(NamedTuple):
    open: float
    high: float
    low: float
    close: float
    settle: float
    volume: float
    oi: float


@dataclass
class Order:
    """A pending order. ``size`` is signed: positive = buy, negative = sell.

    Submitted during the SIGNAL phase of bar ``i``; fills at the OPEN of bar
    ``i + 1`` against that day's active calendar contract for ``symbol``.
    """
    symbol: str
    size: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None       # stop price; ignored for MARKET orders
    reason: Reason = Reason.SIGNAL


@dataclass
class Fill:
    bar_index: int
    date: Date
    symbol: str
    contract: str
    size: int                            # signed: positive = bought, negative = sold
    price: float
    commission: float
    reason: Reason = Reason.SIGNAL
    realized_pnl: float = 0.0            # pnl on the portion of size that reduced a position
