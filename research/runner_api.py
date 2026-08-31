"""Single-window backtest execution with a leak-safe warmup pad.

Every fold, the full-span diagnostic run, and the holdout run all go through
this one function so the "slice + pad + don't record the pad" logic exists
in exactly one place.
"""

from __future__ import annotations

import logging

from core.market import MarketData, build_market_data, slice_market
from datafeed.data_manager import DataManager
from datafeed.products import require_products
from runner import run_single_backtest
from strategies.base import BarContext
from research.splits import Window

logger = logging.getLogger(__name__)


def slice_start(window: Window, pad: int) -> int:
    """First bar (in ``market``'s own numbering) of the slice ``run_window``
    builds for ``window`` at ``pad`` bars of warmup.

    Exported because callers that hand the strategy a *bar-indexed array* --
    ``research.metalabel``'s per-bar ``P(win)`` being the one that exists --
    have to convert between absolute and slice-local numbering with exactly
    the formula ``run_window`` used. A second, independently-drifting copy of
    ``max(0, window.start - pad)`` is precisely the off-by-``lo`` bug that the
    trade-log remapping below already exists to prevent.
    """
    return max(0, window.start - pad)


def load_market(symbols, start: str, end: str, update: bool = False) -> MarketData:
    """Load the full-span ``MarketData`` once, shared read-only across every
    trial/fold -- research code should never re-hit ``DataManager`` per
    trial, only ``slice_market`` this one result.
    """
    resolved = require_products(symbols)
    dm = DataManager(symbols=resolved, update=update)
    universe = dm.get_universe_bundle(start_date=start, end_date=end)
    return build_market_data(universe)


def run_window(
    market: MarketData,
    strategy_cls: type,
    params: dict,
    window: Window,
    *,
    cash: float,
    slippage: float = 0.0,
    pad: int = 0,
    bar_context_cls: type = BarContext,
) -> dict:
    """Run one *isolated* backtest over ``window``: fresh cash, flat
    position, on a slice that starts ``pad`` bars earlier (for indicator
    warmup) but records equity/trades only from ``window.start`` onward
    (via ``Engine(warmup_bars=...)``). The strategy never sees a bar at or
    past ``window.end``.

    Returns ``run_single_backtest``'s dict (``result``, ``metrics``,
    ``engine``) plus ``window`` and ``effective_start``: the bar (relative
    to the full ``market``) at which the strategy's own indicators actually
    became valid, which is also the bar ``equity_records[0]`` corresponds
    to. If this is later than ``window.start``, ``pad`` was too small for
    this parameter set and the window's true evaluation start slipped --
    callers comparing scores across trials/folds should watch for this
    rather than assume every run starts at the same point.

    ``trade_logs``' ``open_bar``/``close_bar`` are remapped back to
    ``market``'s own bar numbering before returning: internally the engine
    only ever sees the ``[lo, window.end)`` slice and knows nothing but
    slice-local indices, so every bar index it produces is off by ``lo``
    unless ``pad == window.start`` (i.e. unless ``lo`` happens to be 0).
    Callers get absolute bars uniformly, regardless of what pad they asked
    for -- ``equity_records`` need no such fix since they carry no numeric
    bar field. Address them by list position against ``effective_start``,
    not against ``window.start``: the two coincide whenever ``pad`` covers
    the strategy's warmup (the intended case), but when it does not the
    engine drops the un-tradeable head bars from the curve as well, leaving
    the list correspondingly shorter.
    """
    lo = slice_start(window, pad)
    sliced = slice_market(market, lo, window.end)
    warmup_bars = window.start - lo
    outcome = run_single_backtest(
        sliced, strategy_cls, params, cash, slippage,
        warmup_bars=warmup_bars, bar_context_cls=bar_context_cls,
    )

    if lo:
        for t in outcome['result']['trade_logs']:
            t['open_bar'] += lo
            if t['close_bar'] is not None:
                t['close_bar'] += lo

    engine = outcome['engine']
    effective_start = lo + engine.record_start
    if effective_start > window.start:
        logger.warning(
            "%s on %s: indicators need %d warmup bars but pad covers only %d "
            "(bars %d-%d) -- evaluation start slipped from %d to %d.",
            strategy_cls.__name__, window.name, engine.warmup_index, pad,
            lo, window.start, window.start, effective_start,
        )

    return {**outcome, 'window': window, 'effective_start': effective_start, 'lo': lo}
