"""Warmup is per product, so a late-listing product cannot hold the rest out.

A product whose history starts late has NaN indicators over the leading bars.
Warmup used to be a single engine-wide index raised to the slowest of those, so
adding one late lister silently deleted every other product's early history --
putting SA (listed 2019-12) in a universe cost the other products 2015-2019.
"""

import numpy as np
import pytest

from core.engine import Engine
from strategies.base import BarContext, SetupContext, Strategy
from tests.conftest import build_market, build_panel

N_BARS = 40
LATE_START = 25          # bar from which the late lister has any data at all


def _panel(symbol, n_bars=N_BARS, first_valid=0):
    """A tradable product whose weighted close is flat 100 from ``first_valid``."""
    close = np.full(n_bars, 100.0)
    weighted = {
        'open': close, 'high': close, 'low': close, 'close': close, 'settle': close,
        'oi': np.full(n_bars, 100.0), 'volume': np.full(n_bars, 100.0),
        'session': np.ones(n_bars),
    }
    code = f'{symbol}509'
    contracts = {code: {
        i: (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
        for i in range(first_valid, n_bars)
    }}
    cbb = [''] * first_valid + [code] * (n_bars - first_valid)
    return build_panel(symbol, n_bars, weighted=weighted, contracts=contracts,
                       contract_by_bar=cbb, first_bar=first_valid)


class _Indicators(Strategy):
    """Registers one indicator per symbol; CF is valid at 0, SA at LATE_START."""

    params = {'lots': 1}

    def setup(self, ctx):
        for sym in ctx.symbols:
            arr = np.full(N_BARS, 1.0)
            if sym == 'SA':
                arr[:LATE_START] = np.nan
            ctx.add_indicator('sig', sym, arr)

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            if ctx.can_trade(sym):
                ctx.set_target(sym, self.p['lots'])


class _IgnoresCanTrade(_Indicators):
    """A strategy that never asks whether a product is tradable."""

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            ctx.set_target(sym, self.p['lots'])


@pytest.fixture
def market():
    return build_market(
        {'CF': _panel('CF'), 'SA': _panel('SA', first_valid=LATE_START)},
        N_BARS,
    )


def _run(market, strategy_cls=_Indicators, **kwargs):
    strategy = strategy_cls()
    eng = Engine(market, strategy, initial_cash=1_000_000.0, **kwargs)
    eng.run_backtest(SetupContext, BarContext)
    return eng


def test_warmup_is_tracked_per_symbol(market):
    strategy = _Indicators()
    eng = Engine(market, strategy, initial_cash=1_000_000.0)
    strategy.setup(SetupContext(eng))

    assert eng.warmup_by_symbol == {'CF': 0, 'SA': LATE_START}
    assert eng.warmup_index == 0                # the first product that is ready
    assert eng.warmup_full == LATE_START        # the last one


def test_late_lister_does_not_delay_the_others(market):
    eng = _run(market)
    fills = {}
    for entry in eng.signal_log:
        fills.setdefault(entry['symbol'], entry['date'])

    dates = list(market.dates)
    assert fills['CF'] == dates[1]                      # traded from the start
    assert fills['SA'] >= dates[LATE_START]              # waited for its own data


def test_orders_for_a_cold_symbol_are_dropped(market):
    """Holds even when the strategy never calls can_trade."""
    eng = _run(market, strategy_cls=_IgnoresCanTrade)
    early = [e['date'] for e in eng.signal_log if e['symbol'] == 'CF']
    late = [e['date'] for e in eng.signal_log if e['symbol'] == 'SA']
    dates = list(market.dates)

    assert early and early[0] == dates[1]
    assert late and min(late) >= dates[LATE_START]


def test_can_trade_reports_a_still_warming_symbol_as_untradable(market):
    strategy = _Indicators()
    eng = Engine(market, strategy, initial_cash=1_000_000.0)
    strategy.setup(SetupContext(eng))

    cold = BarContext(eng, LATE_START - 1, market.dates[LATE_START - 1])
    assert cold.can_trade('CF')
    assert not cold.can_trade('SA')

    warm = BarContext(eng, LATE_START, market.dates[LATE_START])
    assert warm.can_trade('CF')
    assert warm.can_trade('SA')


def test_caller_floor_still_applies_to_every_symbol(market):
    """The research pad must keep the whole universe out, late lister or not."""
    strategy = _Indicators()
    eng = Engine(market, strategy, initial_cash=1_000_000.0, warmup_bars=30)
    strategy.setup(SetupContext(eng))

    assert eng.warmup_by_symbol == {'CF': 30, 'SA': 30}
    assert eng.warmup_index == 30
    assert eng.record_start == 30


# --------------------------------------------------------------------------
# require_warmup -- the one supported way to move warmup_by_symbol
# --------------------------------------------------------------------------

def test_require_warmup_only_ever_raises(market):
    """Indicators register in arbitrary order; the strictest one has to win."""
    eng = Engine(market, _Indicators())
    assert eng.require_warmup('CF', 12) == 12
    assert eng.require_warmup('CF', 30) == 30
    assert eng.require_warmup('CF', 5) == 30      # a laxer one cannot undo it
    assert eng.warmup_by_symbol['CF'] == 30


def test_require_warmup_never_undercuts_the_caller_floor(market):
    """A research pad is a floor, not a suggestion."""
    eng = Engine(market, _Indicators(), warmup_bars=20)
    assert eng.require_warmup('CF', 3) == 20
    assert eng.warmup_by_symbol['CF'] == 20


def test_require_warmup_leaves_the_other_products_alone(market):
    eng = Engine(market, _Indicators())
    eng.require_warmup('SA', 33)
    assert eng.warmup_by_symbol == {'CF': 0, 'SA': 33}
    assert eng.warmup_index == 0        # CF is unaffected and trades from bar 0
    assert eng.warmup_full == 33


def test_require_warmup_starts_an_unknown_symbol_at_the_floor(market):
    """A symbol the market does not carry still respects the pad, not zero."""
    eng = Engine(market, _Indicators(), warmup_bars=15)
    assert eng.require_warmup('ZZ', 4) == 15


def _mixed_universe(n=6):
    """SA ready from bar 0, CF only from bar 3.

    Two products, not one: ``warmup_index`` is the *minimum* across the
    universe, so with a single slow product the signal phase is gated wholesale
    and no order is ever placed to drop. The drop needs a universe where
    something else is already trading -- which is the case that actually
    happens.
    """
    rows = {i: (100.0, 101.0, 99.0, 100.0, 100.0, 0, 10) for i in range(n)}
    panels = {
        sym: build_panel(sym, n, weighted={'session': [1.0] * n},
                         contracts={f'{sym}509': rows}, contract_by_bar=[f'{sym}509'] * n)
        for sym in ('SA', 'CF')
    }
    return build_market(panels, n)


class _LateCf(Strategy):
    """CF's indicator is valid only from bar 3; SA's from bar 0."""

    n_bars = 6
    guard = False

    def setup(self, ctx):
        ctx.add_indicator('ready', 'SA', np.ones(self.n_bars))
        late = np.full(self.n_bars, np.nan)
        late[3:] = 1.0
        ctx.add_indicator('ready', 'CF', late)

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            if self.guard and not ctx.can_trade(sym):
                continue
            ctx.set_target(sym, 1)


def test_orders_dropped_for_warmup_are_counted_and_reported(caplog):
    """The one order the engine refuses that the strategy cannot see. A
    strategy without a ``can_trade`` guard signals through a product's warmup
    and the lots simply vanish -- indistinguishable, from the outside, from a
    signal that never fired. Rejections, deferrals and stranded rolls are all
    disclosed; this one was not.
    """
    import logging

    eng = Engine(_mixed_universe(), _LateCf(), initial_cash=1_000_000.0)
    with caplog.at_level(logging.WARNING, logger='core.engine'):
        out = eng.run_backtest(SetupContext, BarContext)

    assert set(out['warmup_skips']) == {'CF'}          # SA was ready all along
    skips = out['warmup_skips']['CF']
    assert skips['n_orders'] == 3                      # bars 0, 1, 2
    assert (skips['first_bar'], skips['last_bar']) == (0, 2)
    assert skips['ready_at'] == 3
    assert 'can_trade' in caplog.text
    assert eng.broker.net_position('CF') == 1          # bar 3's signal got through


def test_a_guarded_strategy_records_no_warmup_skips():
    strat = _LateCf()
    strat.guard = True
    eng = Engine(_mixed_universe(), strat, initial_cash=1_000_000.0)
    out = eng.run_backtest(SetupContext, BarContext)
    assert out['warmup_skips'] == {}
    assert eng.broker.net_position('CF') == 1
