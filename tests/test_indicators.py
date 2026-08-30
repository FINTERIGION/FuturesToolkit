import numpy as np
import pytest
import talib

from core.engine import Engine
from strategies.base import BarContext, SetupContext, Strategy
from tests.conftest import build_market, build_panel


def test_guard_allows_leading_nan():
    from core.indicators import guard
    arr = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
    out = guard(arr)
    assert list(out[2:]) == [1.0, 2.0, 3.0]


def test_guard_rejects_embedded_nan():
    from core.indicators import guard
    arr = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError):
        guard(arr)


def test_guard_passes_clean_series_through():
    from core.indicators import guard
    arr = np.array([1.0, 2.0, 3.0])
    assert list(guard(arr)) == [1.0, 2.0, 3.0]


class _RecordingStrategy(Strategy):
    def setup(self, ctx):
        self.recorded = []
        close = ctx.close('SA')
        ctx.add_indicator('sma', 'SA', talib.SMA(close, 20))

    def on_bar(self, ctx):
        self.recorded.append(ctx.i)


def test_warmup_skips_bars_before_indicator_is_valid():
    n = 30
    panel = build_panel('SA', n, weighted={'close': list(range(1, n + 1))})
    md = build_market({'SA': panel}, n)
    strat = _RecordingStrategy()
    engine = Engine(md, strat, initial_cash=100_000.0)
    engine.run_backtest(SetupContext, BarContext)

    # SMA(20) is NaN for indices 0..18 (its first 19 leading bars); on_bar
    # should only fire from index 19 onward.
    assert engine.warmup_index == 19
    assert strat.recorded == list(range(19, n))
