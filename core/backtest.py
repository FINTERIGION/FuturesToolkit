"""Run one backtest and score it: engine + metrics, nothing else.

This lives in ``core`` rather than next to the CLI that used to own it
because it is what every other layer actually needs: ``research.runner_api``
calls it once per trial per fold, and ``web.routers.backtest`` calls it per
request. Neither wants argparse, a results directory, or a plotter, and
neither should have to import a top-level script to reach a function with no
CLI in it. That import was also a packaging bug: ``pyproject.toml`` ships
``core``/``research``/``web`` as packages while the CLI stays a loose module
at the repo root, so an installed copy raised ``ModuleNotFoundError`` the
first time any of them was used.
"""

from __future__ import annotations

from core.engine import Engine
from core.market import MarketData
from core.metrics import compute_metrics
from strategies.base import BarContext, SetupContext

__all__ = ['run_single_backtest']


def run_single_backtest(
    market: MarketData,
    strategy_cls: type,
    params: dict,
    cash: float,
    slippage: float = 0.0,
    warmup_bars: int = 0,
) -> dict:
    """Run one backtest with no side effects (no plotting, no file writes).

    Returns ``{'result', 'metrics', 'engine'}``. The engine comes back because
    callers need what only it holds: ``live.signal`` reads ``pending`` and the
    armed brackets off it, and ``research.runner_api`` reads ``record_start``.
    """
    strategy = strategy_cls(**params)
    engine = Engine(market, strategy, initial_cash=cash, slippage=slippage, warmup_bars=warmup_bars)
    result = engine.run_backtest(SetupContext, BarContext)
    metrics = compute_metrics(
        result['equity_records'], result['trade_logs'], cash,
        liquidation_count=result['liquidation_count'],
        rejected_count=len(result['rejections']),
        blown_up=result['blown_up'],
    )
    return {'result': result, 'metrics': metrics, 'engine': engine}
