"""Factor package tests: discovery, the base contract, and every bundled factor.

The parametrized cases run over ``discover_factors()`` rather than a fixed
list, so a new factor dropped into ``factors/`` -- including a private,
gitignored one -- is covered the moment it exists.
"""

import numpy as np
import pytest

import factors
from factors import discover_factors, load_factor, name_for
from factors.base import Factor, FactorContext, compute_factor
from factors.primitives import ts_return
from research.space import check_constraints, resolve_space

from tests.conftest import build_market, build_panel

SYMBOLS = ('SA', 'CF', 'AG')
N_BARS = 300

ALL_FACTORS = sorted(discover_factors().items())


def _market(symbols=SYMBOLS, n_bars=N_BARS, seed=0):
    rng = np.random.default_rng(seed)
    panels = {}
    for offset, sym in enumerate(symbols):
        t = np.arange(n_bars, dtype='float64')
        close = 100.0 + 0.05 * t + 3.0 * np.sin(t / 12.0 + offset) + rng.normal(0, 0.4, n_bars)
        close = np.abs(close) + 1.0
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) + 0.3
        low = np.minimum(open_, close) - 0.3
        oi = rng.uniform(5000, 6000, n_bars)
        volume = rng.uniform(1000, 2000, n_bars)
        weighted = {
            'open': open_, 'high': high, 'low': low, 'close': close,
            'settle': close.copy(), 'oi': oi, 'volume': volume,
            'session': np.ones(n_bars),
        }
        code = f'{sym}2401'
        contracts = {
            code: {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                   for i in range(n_bars)},
        }
        panels[sym] = build_panel(
            sym, n_bars, weighted=weighted, contracts=contracts,
            contract_by_bar=[code] * n_bars, first_bar=0,
        )
    return build_market(panels, n_bars)


class _Constant(Factor):
    """A factor whose ``compute`` returns the same finite score forever."""

    def compute_symbol(self, ctx, sym):
        return np.full(ctx.market.n_bars, 1.0)


class _NoOverride(Factor):
    """A factor that implements neither ``compute`` nor ``compute_symbol``."""


class _WrongShape(Factor):
    def compute_symbol(self, ctx, sym):
        return np.zeros(3)  # deliberately not n_bars


class _AllNaN(Factor):
    def compute_symbol(self, ctx, sym):
        return np.full(ctx.market.n_bars, np.nan)


class _WithInf(Factor):
    def compute_symbol(self, ctx, sym):
        out = np.arange(ctx.market.n_bars, dtype='float64')
        out[5] = np.inf
        out[6] = -np.inf
        return out


class _MidHistoryNaN(Factor):
    def compute_symbol(self, ctx, sym):
        out = np.arange(ctx.market.n_bars, dtype='float64')
        out[10] = np.nan
        return out


class _NegativeDirection(Factor):
    direction = -1

    def compute_symbol(self, ctx, sym):
        return np.arange(ctx.market.n_bars, dtype='float64')


class _BadDirection(Factor):
    direction = 0

    def compute_symbol(self, ctx, sym):
        return np.zeros(ctx.market.n_bars)


class _Longer(Factor):
    params = {'lookback': 5}

    def compute_symbol(self, ctx, sym):
        return ts_return(ctx.close(sym), self.p['lookback'], 0)


# ---------------------------------------------------------------------
# discovery / name_for / load_factor
# ---------------------------------------------------------------------

def test_the_bundled_factors_are_discovered():
    registry = discover_factors()
    assert {'momentum', 'volatility', 'carry', 'trend_accel'} <= set(registry)
    assert all(issubclass(cls, Factor) for cls in registry.values())


def test_name_for_strips_the_factor_suffix_but_keeps_an_empty_stem():
    class CarryFactor(Factor):
        pass

    assert name_for(CarryFactor) == 'carry'

    class _Bare(Factor):
        pass
    _Bare.__name__ = '_factor'  # already-snake name equal to the suffix itself
    assert name_for(_Bare) == '_factor'


def test_load_factor_resolves_a_short_name():
    cls = load_factor('momentum')
    assert issubclass(cls, Factor)


def test_load_factor_resolves_a_dotted_path():
    cls = load_factor(f'{__name__}:_Constant')
    assert cls is _Constant


def test_load_factor_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        load_factor('does_not_exist')


def test_load_factor_rejects_a_non_factor():
    with pytest.raises(TypeError):
        load_factor(f'{__name__}:_market')


# ---------------------------------------------------------------------
# FactorContext
# ---------------------------------------------------------------------

def test_context_panel_matches_the_per_symbol_accessors():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = ctx.panel('close')
    for j, sym in enumerate(ctx.symbols):
        np.testing.assert_array_equal(panel[:, j], ctx.close(sym))


def test_context_panel_rejects_an_unknown_field():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    with pytest.raises(ValueError):
        ctx.panel('bogus')


def test_context_rejects_symbols_the_market_does_not_have():
    market = _market()
    with pytest.raises(ValueError):
        FactorContext(market, list(market.symbols) + ['ZZ'])
    ctx = FactorContext(market, market.symbols)
    with pytest.raises(ValueError):
        ctx.close('ZZ')


# ---------------------------------------------------------------------
# Factor / compute_factor base contract
# ---------------------------------------------------------------------

def test_factor_produces_a_well_formed_panel():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(_Constant(), ctx)
    assert panel.values.shape == (market.n_bars, len(market.symbols))
    assert panel.tradable.shape == panel.values.shape
    assert panel.symbols == list(market.symbols)


def test_factor_has_a_resolvable_search_space():
    space = resolve_space(_Longer)
    assert 'lookback' in space


def test_factor_direction_is_a_sign():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(_Constant(), ctx)
    assert panel.direction in (1, -1)


def test_param_overrides_change_the_output():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    p1 = compute_factor(_Longer(lookback=5), ctx)
    p2 = compute_factor(_Longer(lookback=20), ctx)
    assert not np.allclose(np.nan_to_num(p1.values), np.nan_to_num(p2.values))


def test_a_longer_lookback_starts_scoring_later():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    short = compute_factor(_Longer(lookback=5), ctx)
    long_ = compute_factor(_Longer(lookback=50), ctx)
    first_short = int(np.argmax(np.isfinite(short.values[:, 0])))
    first_long = int(np.argmax(np.isfinite(long_.values[:, 0])))
    assert first_long > first_short


def test_compute_stacks_compute_symbol_in_symbol_order():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    factor = _Constant()
    stacked = factor.compute(ctx)
    assert list(stacked) == list(ctx.symbols)
    for sym in ctx.symbols:
        np.testing.assert_array_equal(stacked[sym], factor.compute_symbol(ctx, sym))


def test_a_factor_implementing_neither_method_says_so():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    with pytest.raises(NotImplementedError):
        compute_factor(_NoOverride(), ctx)


def test_direction_minus_one_flips_the_sign():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(_NegativeDirection(), ctx)
    raw = np.arange(market.n_bars, dtype='float64')
    np.testing.assert_array_equal(panel.values[:, 0], -raw)


def test_a_bad_direction_is_rejected():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    with pytest.raises(ValueError):
        compute_factor(_BadDirection(), ctx)


def test_a_wrong_shape_is_rejected():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    with pytest.raises(ValueError):
        compute_factor(_WrongShape(), ctx)


def test_an_all_nan_factor_warns_but_still_returns_a_panel(caplog):
    market = _market()
    ctx = FactorContext(market, market.symbols)
    with caplog.at_level('WARNING'):
        panel = compute_factor(_AllNaN(), ctx)
    assert panel.values.shape == (market.n_bars, len(market.symbols))
    assert np.isnan(panel.values).all()
    assert any('all-NaN' in rec.message for rec in caplog.records)


def test_infinities_become_nan():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(_WithInf(), ctx)
    assert np.isnan(panel.values[5, 0])
    assert np.isnan(panel.values[6, 0])


def test_a_mid_history_nan_survives():
    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(_MidHistoryNaN(), ctx)
    assert np.isnan(panel.values[10, 0])
    assert np.isfinite(panel.values[9, 0])
    assert np.isfinite(panel.values[11, 0])


def test_tradable_requires_a_session_and_a_contract():
    panel = build_panel(
        'SA', 10,
        weighted={'session': np.array([1., 1., 0., 1., 1., 1., 1., 1., 1., 1.])},
        contracts={'SA2401': {i: (1., 1., 1., 1., 1., 100., 10.) for i in range(1, 10)}},
        contract_by_bar=[''] + ['SA2401'] * 9,
        first_bar=0,
    )
    market = build_market({'SA': panel}, 10)
    ctx = FactorContext(market, ['SA'])
    panel_out = compute_factor(_Constant(), ctx)
    tradable = panel_out.tradable[:, 0]
    assert not tradable[0]   # no contract yet
    assert not tradable[2]   # session == 0
    assert tradable[1] and tradable[9]


# ---------------------------------------------------------------------
# every bundled factor, parametrized
# ---------------------------------------------------------------------

@pytest.mark.parametrize('name,cls', ALL_FACTORS)
def test_bundled_factor_has_a_resolvable_space_or_is_fully_fixed(name, cls):
    fixed = set(getattr(cls, 'fixed_params', ()) or ())
    if set(cls.params or {}) <= fixed:
        pytest.skip(f'{name}: every param is fixed by design')
    space = resolve_space(cls)
    assert space


@pytest.mark.parametrize('name,cls', ALL_FACTORS)
def test_bundled_factor_produces_a_well_formed_panel(name, cls):
    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(cls(), ctx)
    assert panel.values.shape == (market.n_bars, len(market.symbols))
    assert panel.direction in (1, -1)


@pytest.mark.parametrize('name,cls', ALL_FACTORS)
def test_bundled_factor_constraints_hold_at_defaults(name, cls):
    assert check_constraints(cls, dict(cls.params or {}))


# ---------------------------------------------------------------------
# trend_accel: identical to FinterMomentumStrategy's trend_accel
# ---------------------------------------------------------------------

def test_trend_accel_matches_the_one_bar_ma_spread_change():
    import talib

    from factors.primitives import ts_diff
    from factors.trend_accel import TrendAccelFactor

    market = _market()
    ctx = FactorContext(market, market.symbols)
    panel = compute_factor(TrendAccelFactor(), ctx)

    close = ctx.close('SA')
    bias = talib.SMA(close, 7) - talib.SMA(close, 20)
    np.testing.assert_allclose(panel.values[:, 0], ts_diff(bias, 1), equal_nan=True)


def test_trend_accel_a_slower_leg_starts_scoring_later():
    from factors.trend_accel import TrendAccelFactor

    market = _market()
    ctx = FactorContext(market, market.symbols)
    short = compute_factor(TrendAccelFactor(fast_period=3, slow_period=15), ctx)
    long_ = compute_factor(TrendAccelFactor(fast_period=3, slow_period=40), ctx)
    first_short = int(np.argmax(np.isfinite(short.values[:, 0])))
    first_long = int(np.argmax(np.isfinite(long_.values[:, 0])))
    assert first_long > first_short
