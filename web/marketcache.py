"""LRU cache over ``research.runner_api.load_market``.

``load_market`` re-reads every symbol's CSV from disk on every call, which is
fine for a CLI run (one call, then the process exits) but not for a panel
where a user tries several strategies against the same universe/date range
back to back. ``MarketData`` is documented as a read-only, shareable object
(see ``research/runner_api.py``), so caching it here is safe for the same
reason it is safe to share across one validation run's walk-forward folds.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Tuple

from core.market import MarketData
from research.runner_api import load_market

from web.config import MARKET_CACHE_SIZE

logger = logging.getLogger('futurestoolkit.web')

_Key = Tuple[Tuple[str, ...], str, str]


class MarketCache:
    def __init__(self, maxsize: int = MARKET_CACHE_SIZE):
        self._maxsize = maxsize
        self._store: "OrderedDict[_Key, MarketData]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(symbols, start: str, end: str) -> _Key:
        """Symbols in the order given -- deliberately not sorted.

        ``MarketData`` keeps its products in load order and the engine walks
        them in that order, which decides who gets margin first when it binds.
        A sorted key handed ``[CF, SA]`` whatever ``[SA, CF]`` had cached, so
        one request could backtest differently depending on what ran before
        it. Keyed on order, a request always gets the order it asked for --
        the same answer ``ft.py backtest`` gives for that ``--symbols``.
        """
        return (tuple(symbols), str(start), str(end))

    def get(self, symbols, start: str, end: str, update: bool = False) -> MarketData:
        key = self._key(symbols, start, end)
        if not update:
            with self._lock:
                hit = self._store.get(key)
                if hit is not None:
                    self._store.move_to_end(key)
                    return hit

        market = load_market(symbols, start, end, update=update)
        with self._lock:
            self._store[key] = market
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug('MarketCache: evicted %s', evicted_key)
        return market

    def invalidate(self) -> None:
        """Drop everything -- called after a registry write or a data update,
        either of which can change what a cached ``MarketData`` reflects."""
        with self._lock:
            self._store.clear()


cache = MarketCache()
