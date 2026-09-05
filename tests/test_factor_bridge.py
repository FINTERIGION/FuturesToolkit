"""Factor-to-strategy bridge tests.

The bridge's job is that writing a factor is enough to get a runnable,
tunable strategy. These tests check the generated class is well-formed, is
found by the normal strategy discovery, and actually trades.
"""

import numpy as np
import pytest

from core.params import Int
from factors.base import Factor
from research.space import resolve_space
from strategies import discover_strategies, load_strategy
from strategies.cross_section import CrossSectionMixin, DEFAULT_PARAMS
from strategies.factor_bridge import FACTOR_STRATEGIES, make_factor_strategy
from core.backtest import run_single_backtest

from tests.conftest import build_market, build_panel

SYMBOLS = ('SA', 'CF', 'AG', 'C', 'JM')
N_BARS = 400


def _market(n_bars=N_BARS, symbols=SYMBOLS, seed=0):
    rng = np.random.default_rng(seed)
    panels = {}
    for offset, sym in enumerate(symbols):
        t = np.arange(n_bars, dtype='float64')
        close = 100.0 + 0.04 * t + 4.0 * np.sin(t / 15.0 + offset) + rng.normal(0, 0.3, n_bars)
        close = np.abs(close) + 10.0
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) + 0.5
        low = np.minimum(open_, close) - 0.5
        oi = rng.uniform(5000, 6000, n_bars)
        volume = rng.uniform(1000, 2000, n_bars)
        weighted = {
            'open': open_, 'high': high, 'low': low, 'close': close,
            'settle': close.copy(), 'oi': oi, 'volume': volume, 'session': np.ones(n_bars),
        }
        code = f'{sym}C1'
        contracts = {code: {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                             for i in range(n_bars)}}
        panels[sym] = build_panel(sym, n_bars, weighted=weighted, contracts=contracts,
                                   contract_by_bar=[code] * n_bars, first_bar=0)
    return build_market(panels, n_bars)


class _SimpleFactor(Factor):
    params = {'lookback': 20}
    space = {'lookback': Int(5, 60)}
    direction = 1

    def compute_symbol(self, ctx, sym):
        from factors.primitives import ts_return
        return ts_return(ctx.close(sym), self.p['lookback'], 0)


class _NegativeFactor(_SimpleFactor):
    direction = -1


class _ClashingFactor(Factor):
    """Declares a param name that collides with the trading rule's own."""
    params = {'top_k': 99}


# ---------------------------------------------------------------------
# make_factor_strategy
# ---------------------------------------------------------------------

def test_one_strategy_is_generated_per_factor():
    from factors import discover_factors
    assert set(FACTOR_STRATEGIES) == {f'factor_{name}' for name in discover_factors()}


def test_generated_strategies_are_found_by_normal_discovery():
    registry = discover_strategies()
    assert 'factor_momentum' in registry
    assert 'factor_trend_accel' in registry
    assert registry['factor_momentum'] is FACTOR_STRATEGIES['factor_momentum']


def test_the_abstract_helpers_are_not_offered_as_strategies():
    registry = discover_strategies()
    assert 'cross_section_mixin' not in registry
    assert not any(name.endswith('factor_bridge') for name in registry)


def test_generated_class_names_read_as_strategies():
    cls = make_factor_strategy(_SimpleFactor, 'simple')
    assert cls.__name__ == 'FactorSimpleStrategy'
    from strategies import name_for
    assert name_for(cls) == 'factor_simple'


def test_generated_params_merge_the_factor_and_the_trading_rule():
    cls = make_factor_strategy(_SimpleFactor, 'simple')
    assert cls.params['lookback'] == 20
    for key, value in DEFAULT_PARAMS.items():
        assert cls.params[key] == value


def test_generated_space_covers_both_halves():
    cls = make_factor_strategy(_SimpleFactor, 'simple')
    space = resolve_space(cls)
    assert 'lookback' in space
    assert 'top_k' in space
    assert 'rebalance_days' in space


def test_risk_limits_are_never_tuned():
    cls = make_factor_strategy(_SimpleFactor, 'simple')
    space = resolve_space(cls)
    for key in ('risk_budget', 'max_gross_margin', 'min_universe'):
        assert key not in space
        assert key in cls.fixed_params


def test_factor_constraints_are_carried_over():
    class _Constrained(Factor):
        params = {'fast': 5, 'slow': 20}
        constraints = (lambda p: p['fast'] < p['slow'],)

    cls = make_factor_strategy(_Constrained, 'constrained')
    assert cls.constraints == _Constrained.constraints


def test_a_param_name_clash_is_an_error_not_a_silent_overwrite():
    with pytest.raises(ValueError):
        make_factor_strategy(_ClashingFactor, 'clashing')


def test_a_clashing_factor_is_skipped_rather_than_breaking_discovery(caplog):
    import strategies.factor_bridge as bridge_mod

    class _FakeRegistry(dict):
        pass

    fake = {'ok': _SimpleFactor, 'clashing': _ClashingFactor}
    orig = bridge_mod.discover_factors
    bridge_mod.discover_factors = lambda: fake
    try:
        with caplog.at_level('WARNING'):
            generated = bridge_mod._generate_all()
    finally:
        bridge_mod.discover_factors = orig
    assert 'factor_ok' in generated
    assert 'factor_clashing' not in generated
    assert any('clashing' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------
# end-to-end trading
# ---------------------------------------------------------------------

def test_a_bridged_strategy_runs_and_trades():
    cls = make_factor_strategy(_SimpleFactor, 'simple')
    market = _market()
    outcome = run_single_backtest(market, cls, {}, cash=1_000_000.0, slippage=0.0)
    assert outcome['result']['trade_logs']
    assert outcome['metrics']['n_trades'] > 0


def test_a_negative_direction_factor_trades_the_opposite_side():
    pos_cls = make_factor_strategy(_SimpleFactor, 'simple_pos')
    neg_cls = make_factor_strategy(_NegativeFactor, 'simple_neg')
    market = _market()
    pos_out = run_single_backtest(market, pos_cls, {}, cash=1_000_000.0, slippage=0.0)
    neg_out = run_single_backtest(market, neg_cls, {}, cash=1_000_000.0, slippage=0.0)

    def _first_direction(trades):
        return trades[0]['direction'] if trades else None

    pos_trades = sorted(pos_out['result']['trade_logs'], key=lambda t: t['open_bar'])
    neg_trades = sorted(neg_out['result']['trade_logs'], key=lambda t: t['open_bar'])
    assert pos_trades and neg_trades
    # Same universe, same ranking magnitude, opposite sign -> the symbol that
    # was long under +1 direction should be short (or vice versa) under -1,
    # for at least the very first rebalance.
    pos_first = {t['symbol']: t['direction'] for t in pos_trades if t['open_bar'] == pos_trades[0]['open_bar']}
    neg_first = {t['symbol']: t['direction'] for t in neg_trades if t['open_bar'] == neg_trades[0]['open_bar']}
    shared = set(pos_first) & set(neg_first)
    assert shared
    assert any(pos_first[s] != neg_first[s] for s in shared)


def test_a_factor_scoring_nothing_never_trades():
    class _NoScore(Factor):
        def compute_symbol(self, ctx, sym):
            return np.full(ctx.market.n_bars, np.nan)

    cls = make_factor_strategy(_NoScore, 'noscore')
    market = _market()
    outcome = run_single_backtest(market, cls, {}, cash=1_000_000.0, slippage=0.0)
    assert outcome['result']['trade_logs'] == []


def test_the_generated_strategy_uses_the_shared_cross_section_rule():
    cls = make_factor_strategy(_SimpleFactor, 'simple')
    assert issubclass(cls, CrossSectionMixin)


def test_the_bridge_reproduces_the_hand_written_cross_sectional_strategy():
    """A bridged momentum factor and the hand-written
    ``CrossSectionalMomentumStrategy`` apply the identical trading rule
    (``CrossSectionMixin``); with the momentum score aligned to the same
    formula, both should select the same longs/shorts on the same universe."""
    from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
    from factors.momentum import MomentumFactor

    bridged_cls = make_factor_strategy(MomentumFactor, 'momentum')
    market = _market()

    shared = dict(top_k=2, rebalance_days=5, atr_period=20,
                  risk_budget=0.01, max_gross_margin=0.6, min_universe=4)
    bridged_params = {**shared, 'lookback': 60, 'skip': 5}
    hand_params = {**shared, 'lookback': 60, 'skip': 5}

    bridged_out = run_single_backtest(market, bridged_cls, bridged_params, cash=1_000_000.0, slippage=0.0)
    hand_out = run_single_backtest(market, CrossSectionalMomentumStrategy, hand_params, cash=1_000_000.0, slippage=0.0)

    def _first_selection(trades):
        if not trades:
            return {}
        first_bar = min(t['open_bar'] for t in trades)
        return {t['symbol']: t['direction'] for t in trades if t['open_bar'] == first_bar}

    assert _first_selection(bridged_out['result']['trade_logs']) == \
        _first_selection(hand_out['result']['trade_logs'])
