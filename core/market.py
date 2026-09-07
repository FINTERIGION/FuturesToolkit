"""Market data: per-product weighted series and per-contract lifecycles.

Built from ``datafeed.data_manager.DataManager.get_universe_bundle()``. Each
contract stores only the bars where it actually printed a real exchange
quote, not padded to the full union calendar -- a 63-contract x 1610-bar
union calendar is ~85% padding; storing only real rows is the point of this
rewrite (see docs/rewrite-plan.md §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OHLCV_COLS = ['open', 'high', 'low', 'close', 'settle', 'oi', 'volume']


@dataclass
class ContractSeries:
    code: str
    start: int                  # bar index of the first live print
    live_bars: np.ndarray       # int64[n_live], strictly ascending
    ohlcv: np.ndarray           # float64[n_live, 7]: open high low close settle oi volume

    def row_at(self, bar_index: int) -> Optional[np.ndarray]:
        """This contract's OHLCV row for ``bar_index``, or None if it did not print."""
        pos = int(np.searchsorted(self.live_bars, bar_index))
        if pos < len(self.live_bars) and self.live_bars[pos] == bar_index:
            return self.ohlcv[pos]
        return None


@dataclass
class ProductPanel:
    symbol: str
    weighted: Dict[str, np.ndarray]     # 'open'..'volume','session', each float64[n_bars]
    contracts: Dict[str, ContractSeries]
    contract_by_bar: np.ndarray         # object[n_bars]; '' where no contract has a live print
    first_bar: int                      # bar index of the product's listing date

    def is_listed(self, i: int) -> bool:
        return i >= self.first_bar

    def has_session(self, i: int) -> bool:
        return bool(self.weighted['session'][i])

    def active_contract(self, i: int) -> str:
        return self.contract_by_bar[i] or ''

    def can_trade(self, i: int) -> bool:
        return self.is_listed(i) and self.has_session(i) and bool(self.active_contract(i))


@dataclass
class MarketData:
    dates: np.ndarray                   # datetime64[D][n_bars]
    products: Dict[str, ProductPanel]

    @property
    def symbols(self) -> List[str]:
        return list(self.products)

    @property
    def n_bars(self) -> int:
        return len(self.dates)


def slice_market(market: MarketData, start: int, end: int) -> MarketData:
    """Bar-index slice ``[start, end)`` of ``market``, re-based so bar 0 of
    the result is bar ``start`` of the original -- the result behaves like a
    standalone :class:`MarketData` for everything downstream (``Engine``,
    strategies, indicators).

    Used to carve out train/valid/holdout windows for parameter optimization
    and walk-forward validation without ever handing a strategy a numpy view
    that reaches past ``end``.
    """
    start = max(0, start)
    end = min(market.n_bars, end)
    dates = market.dates[start:end]

    products: Dict[str, ProductPanel] = {}
    for symbol, panel in market.products.items():
        weighted = {field: arr[start:end] for field, arr in panel.weighted.items()}
        contract_by_bar = panel.contract_by_bar[start:end]

        contracts: Dict[str, ContractSeries] = {}
        for code, series in panel.contracts.items():
            mask = (series.live_bars >= start) & (series.live_bars < end)
            if not mask.any():
                continue
            live_bars = series.live_bars[mask] - start
            ohlcv = series.ohlcv[mask]
            contracts[code] = ContractSeries(
                code=code, start=int(live_bars[0]), live_bars=live_bars, ohlcv=ohlcv,
            )

        products[symbol] = ProductPanel(
            symbol=symbol,
            weighted=weighted,
            contracts=contracts,
            contract_by_bar=contract_by_bar,
            first_bar=max(0, panel.first_bar - start),
        )

    return MarketData(dates=dates, products=products)


def build_market_data(universe: dict) -> MarketData:
    """Build a :class:`MarketData` from ``DataManager.get_universe_bundle()``."""
    calendar: pd.DatetimeIndex = universe['calendar']
    dates = calendar.values.astype('datetime64[D]')

    products: Dict[str, ProductPanel] = {}
    for symbol in universe['symbols']:
        bundle = universe['products'][symbol]
        products[symbol] = _build_panel(symbol, bundle, calendar)

    return MarketData(dates=dates, products=products)


def _build_panel(symbol: str, bundle: dict, calendar: pd.DatetimeIndex) -> ProductPanel:
    weighted_df = bundle['weighted_df']
    weighted = {
        col: weighted_df[col].to_numpy(dtype='float64')
        for col in ('open', 'high', 'low', 'close', 'settle', 'oi', 'volume', 'session')
    }

    contracts: Dict[str, ContractSeries] = {}
    for code, frame in bundle['contract_frames'].items():
        idx = calendar.get_indexer(frame.index)
        keep = idx >= 0
        idx = idx[keep]
        if len(idx) == 0:
            continue
        order = np.argsort(idx)
        live_bars = idx[order].astype('int64')
        ohlcv = frame[OHLCV_COLS].to_numpy(dtype='float64')[keep][order]
        contracts[code] = ContractSeries(
            code=code, start=int(live_bars[0]), live_bars=live_bars, ohlcv=ohlcv,
        )

    exec_price_df = bundle['exec_price_df']
    contract_by_bar = exec_price_df['contract'].reindex(calendar).fillna('').to_numpy(dtype=object)

    first_print = pd.Timestamp(bundle['first_print'])
    first_bar = int(calendar.searchsorted(first_print))

    return ProductPanel(
        symbol=symbol,
        weighted=weighted,
        contracts=contracts,
        contract_by_bar=contract_by_bar,
        first_bar=first_bar,
    )
