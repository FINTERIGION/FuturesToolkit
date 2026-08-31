"""Cross-sectional momentum strategy tests on a synthetic multi-product market.

Five products with constant per-bar growth rates (SA strongest up, TA
strongest down, FG/CF/MA in between) give a momentum ranking that is exactly
constant from the first valid bar onward -- ``close[t]/close[t-lookback]``
for a geometric series depends only on the growth rate, not on ``t``. That
determinism is what lets these tests assert exact long/short/flat outcomes
without hand-tuning tolerances.
"""

import numpy as np
import pandas as pd

from core.engine import Engine
from strategies.base import BarContext, SetupContext
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from tests.conftest import build_market, build_panel

_GROWTH = {'SA': 0.02, 'FG': 0.01, 'CF': 0.0, 'MA': -0.01, 'TA': -0.02}
_PARAMS = {
    'lookback': 10, 'skip': 0, 'atr_period': 5, 'top_k': 1,
    'rebalance_days': 5, 'risk_budget': 0.5, 'max_gross_margin': 0.9,
    'min_universe': 4,
}


def _trend_panel(symbol, n_bars, rate, tradable=True, base=100.0):
    close = base * (1.0 + rate) ** np.arange(n_bars)
    high, low = close * 1.01, close * 0.99
    weighted = {
        'open': close, 'high': high, 'low': low, 'close': close, 'settle': close,
        'oi': np.full(n_bars, 100.0), 'volume': np.full(n_bars, 100.0),
        'session': np.full(n_bars, 1.0 if tradable else 0.0),
    }
    code = f'{symbol}509'
    contracts = {code: {
        i: (close[i], high[i], low[i], close[i], close[i], 100.0, 100.0) for i in range(n_bars)
    }}
    return build_panel(symbol, n_bars, weighted=weighted, contracts=contracts,
                        contract_by_bar=[code] * n_bars, first_bar=0)


def _build_universe(n_bars=40, untradable=()):
    products = {
        sym: _trend_panel(sym, n_bars, rate, tradable=sym not in untradable)
        for sym, rate in _GROWTH.items()
    }
    return build_market(products, n_bars)


def _run(market, strategy_cls=CrossSectionalMomentumStrategy, **overrides):
    params = {**_PARAMS, **overrides}
    strategy = strategy_cls(**params)
    eng = Engine(market, strategy, initial_cash=100_000.0)
    eng.run_backtest(SetupContext, BarContext)
    return eng, strategy


def test_ranks_and_takes_both_sides():
    eng, _ = _run(_build_universe())
    assert eng.broker.net_position('SA') > 0     # strongest uptrend -> long
    assert eng.broker.net_position('TA') < 0     # strongest downtrend -> short
    for sym in ('FG', 'CF', 'MA'):
        assert eng.broker.net_position(sym) == 0


class _TrackingStrategy(CrossSectionalMomentumStrategy):
    """Records every bar where a rebalance decision actually ran."""

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.decision_bars = []

    def on_bar(self, ctx):
        before = self._next_rebalance
        super().on_bar(ctx)
        if self._next_rebalance != before:
            self.decision_bars.append(ctx.i)


def test_holds_between_rebalances():
    eng, strat = _run(_build_universe(), strategy_cls=_TrackingStrategy)
    assert strat.decision_bars   # sanity: at least one rebalance actually ran

    date_to_bar = {pd.Timestamp(d).date(): i for i, d in enumerate(eng.market.dates)}
    fill_bars = {date_to_bar[entry['date']] for entry in eng.signal_log}
    allowed = {b + 1 for b in strat.decision_bars}
    assert fill_bars <= allowed   # every fill traces back to the bar right after a decision


def test_legs_share_margin_budget():
    eng, _ = _run(_build_universe())
    entry_date = min(t['open_date'] for t in eng.ledger.trades)
    record = next(r for r in eng.equity_records if r['date'] == entry_date)

    assert record['margin_used'] <= 100_000.0 * _PARAMS['max_gross_margin'] * 1.1
    assert eng.broker.liquidation_count == 0


def test_skips_untradable_symbol():
    eng, _ = _run(_build_universe(untradable={'CF'}))
    assert eng.broker.net_position('CF') == 0
    assert 'CF' not in {entry['symbol'] for entry in eng.signal_log}
