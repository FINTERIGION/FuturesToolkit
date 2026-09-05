"""Factor evaluation tests.

The lag tests are the point of this file. Everything else in the factor stack
can be wrong and still produce a plausible-looking report; a one-bar offset in
``forward_returns`` produces a *great*-looking report, because the factor is
being scored against a move it already knew about. These tests pin the offset
down in both directions.
"""

import numpy as np
import pytest

from factors.base import Factor, FactorContext, compute_factor
from research.factor_eval import (
    coverage_summary, cumulative_curves, factor_autocorr, factor_turnover,
    forward_returns, ic_decay, ic_series, ic_summary, panel_forward_returns,
    quantile_returns, annual_slices,
)

from tests.conftest import build_market, build_panel

SYMBOLS = ('SA', 'CF', 'AG', 'C', 'JM')
N_SYMBOLS = len(SYMBOLS)
N_BARS = 400


def _prices(n_bars, symbols, seed=0):
    """One clean, strictly-positive random-walk price series per symbol."""
    rng = np.random.default_rng(seed)
    out = {}
    for sym in symbols:
        steps = rng.normal(0, 1.0, n_bars)
        close = 100.0 + np.cumsum(steps)
        close = np.abs(close) + 10.0
        out[sym] = close
    return out


def _market(n_bars=N_BARS, symbols=SYMBOLS, seed=0, dark_bars=None, roll_at=None):
    """Synthetic multi-product market with real, distinct daily dates (needed
    for ``annual_slices``) and every bar tradable unless ``dark_bars`` marks
    some symbol's sessions off, or ``roll_at`` splits a symbol's contract in two.
    """
    closes = _prices(n_bars, symbols, seed)
    panels = {}
    for sym in symbols:
        close = closes[sym]
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) + 0.5
        low = np.minimum(open_, close) - 0.5
        oi = np.full(n_bars, 5000.0)
        volume = np.full(n_bars, 1000.0)
        session = np.ones(n_bars)
        if dark_bars and sym in dark_bars:
            for i in dark_bars[sym]:
                session[i] = 0.0

        weighted = {
            'open': open_, 'high': high, 'low': low, 'close': close,
            'settle': close.copy(), 'oi': oi, 'volume': volume, 'session': session,
        }

        if roll_at and sym in roll_at:
            split = roll_at[sym]
            code_a, code_b = f'{sym}A', f'{sym}B'
            contracts = {
                code_a: {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                         for i in range(0, split)},
                code_b: {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                         for i in range(split, n_bars)},
            }
            contract_by_bar = [code_a] * split + [code_b] * (n_bars - split)
        else:
            code = f'{sym}C1'
            contracts = {
                code: {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                       for i in range(n_bars)},
            }
            contract_by_bar = [code] * n_bars

        if dark_bars and sym in dark_bars:
            for i in dark_bars[sym]:
                contract_by_bar[i] = ''

        panels[sym] = build_panel(
            sym, n_bars, weighted=weighted, contracts=contracts,
            contract_by_bar=contract_by_bar, first_bar=0,
        )

    dates = np.array(
        [np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(n_bars)],
        dtype='datetime64[D]',
    )
    from core.market import MarketData
    return MarketData(dates=dates, products=panels)


class _CheatFactor(Factor):
    """Computes exactly the lag=1, horizon=1 forward return -- an
    intentional lookahead bug used to prove the evaluation stack does not
    accidentally reward a factor scored against the wrong horizon."""

    def compute_symbol(self, ctx, sym):
        close = ctx.close(sym)
        out = np.full_like(close, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[:-2] = close[2:] / close[1:-1] - 1.0
        return out


class _RandomFactor(Factor):
    def compute_symbol(self, ctx, sym):
        rng = np.random.default_rng(abs(hash(sym)) % (2 ** 32))
        return rng.normal(0, 1, ctx.market.n_bars)


def _panel_of(factor, market, symbols=SYMBOLS):
    ctx = FactorContext(market, symbols)
    return compute_factor(factor, ctx)


# ---------------------------------------------------------------------
# forward_returns: the lag convention
# ---------------------------------------------------------------------

def test_forward_returns_lag_shifts_the_measurement_window():
    market = _market()
    fwd_lag1 = forward_returns(market, [1], lag=1)[1]
    fwd_lag0 = forward_returns(market, [1], lag=0)[1]
    close = market.products['SA'].weighted['close']
    expected_lag1 = close[2] / close[1] - 1.0
    expected_lag0 = close[1] / close[0] - 1.0
    j = list(market.symbols).index('SA')
    assert fwd_lag1[0, j] == pytest.approx(expected_lag1)
    assert fwd_lag0[0, j] == pytest.approx(expected_lag0)


def test_forward_returns_rejects_a_negative_lag():
    market = _market()
    with pytest.raises(ValueError):
        forward_returns(market, [1], lag=-1)


def test_unknown_source_is_rejected():
    market = _market()
    with pytest.raises(ValueError):
        forward_returns(market, [1], source='bogus')


def test_dark_bars_contribute_no_observations():
    market = _market(dark_bars={'SA': [50]})
    fwd = forward_returns(market, [1], lag=1)[1]
    j = list(market.symbols).index('SA')
    # bar 49: entry = bar 50 (dark) -> NaN. bar 48: exit = bar 50 (dark) -> NaN.
    assert np.isnan(fwd[49, j])
    assert np.isnan(fwd[48, j])
    # bar 50 itself trades bars 51->52, neither of which is dark -> finite.
    assert np.isfinite(fwd[50, j])


def test_unlisted_products_are_excluded_until_they_list():
    market = _market()
    market.products['SA'].first_bar = 20
    fwd = forward_returns(market, [1], lag=1)[1]
    j = list(market.symbols).index('SA')
    assert np.isnan(fwd[:19, j]).all()


def test_exec_source_drops_windows_that_span_a_roll():
    market = _market(roll_at={'SA': 100})
    fwd_exec = forward_returns(market, [1], lag=1, source='exec')[1]
    j = list(market.symbols).index('SA')
    # bar 98 -> entry 99, exit 100: exit lands on the new contract -> NaN
    assert np.isnan(fwd_exec[98, j])
    # bar 50 -> entry 51, exit 52: both on contract A -> finite
    assert np.isfinite(fwd_exec[50, j])


# ---------------------------------------------------------------------
# IC: the sentinel tests
# ---------------------------------------------------------------------

def test_ic_is_one_only_at_the_lag_the_factor_was_built_for():
    market = _market()
    factor = _CheatFactor()  # scores close[t+1] - close[t] on bar t
    panel = _panel_of(factor, market)

    fwd_matching = panel_forward_returns(panel, market, [1])[1]  # lag=1 default
    ic_matching = ic_series(panel, fwd_matching)
    summary_matching = ic_summary(ic_matching)
    assert summary_matching['mean'] == pytest.approx(1.0, abs=1e-9)

    fwd_wrong = forward_returns(market, [1], lag=0, symbols=panel.symbols)[1]
    ic_wrong = ic_series(panel, fwd_wrong)
    summary_wrong = ic_summary(ic_wrong)
    assert summary_wrong['mean'] < 0.9


def test_random_factor_has_no_information():
    market = _market(n_bars=800)
    factor = _RandomFactor()
    panel = _panel_of(factor, market)
    fwd = panel_forward_returns(panel, market, [1])[1]
    summary = ic_summary(ic_series(panel, fwd))
    assert abs(summary['mean']) < 0.15


def test_negative_direction_flips_the_sign_of_the_ic():
    market = _market()

    class _Positive(Factor):
        direction = 1

        def compute_symbol(self, ctx, sym):
            close = ctx.close(sym)
            out = np.full_like(close, np.nan)
            out[:-1] = close[1:] - close[:-1]
            return out

    class _Negative(_Positive):
        direction = -1

    pos_panel = _panel_of(_Positive(), market)
    neg_panel = _panel_of(_Negative(), market)
    fwd = panel_forward_returns(pos_panel, market, [1])[1]
    ic_pos = ic_summary(ic_series(pos_panel, fwd))
    ic_neg = ic_summary(ic_series(neg_panel, fwd))
    assert ic_pos['mean'] == pytest.approx(-ic_neg['mean'], abs=1e-9)


def test_ic_decay_covers_every_horizon():
    market = _market()
    factor = _CheatFactor()
    panel = _panel_of(factor, market)
    fwds = forward_returns(market, [1, 5, 10], symbols=panel.symbols)
    decay = ic_decay(panel, fwds)
    assert set(decay) == {1, 5, 10}
    for h, summary in decay.items():
        assert 'mean' in summary and 'ir' in summary


# ---------------------------------------------------------------------
# quantile buckets
# ---------------------------------------------------------------------

def test_quantile_buckets_are_monotone_for_a_perfect_factor():
    market = _market()
    factor = _CheatFactor()
    panel = _panel_of(factor, market)
    fwd = panel_forward_returns(panel, market, [1])[1]
    result = quantile_returns(panel, fwd, n_groups=3, horizon=1)
    assert result['monotonicity'] == pytest.approx(1.0, abs=1e-9)
    means = [result['groups'][g]['mean'] for g in range(3)]
    assert means[0] < means[1] < means[2]


def test_quantile_buckets_are_flat_for_a_random_factor():
    market = _market(n_bars=800)
    factor = _RandomFactor()
    panel = _panel_of(factor, market)
    fwd = panel_forward_returns(panel, market, [1])[1]
    result = quantile_returns(panel, fwd, n_groups=3, horizon=1)
    spread_t = result['spread']['t_stat']
    assert np.isnan(spread_t) or abs(spread_t) < 3.0


# ---------------------------------------------------------------------
# cumulative curves
# ---------------------------------------------------------------------

def test_cumulative_curves_compound_from_one():
    group_returns = np.array([[0.1, -0.1], [0.1, -0.1], [0.1, -0.1]])
    rows, curves = cumulative_curves(group_returns, horizon=1)
    assert curves[0, 0] == pytest.approx(1.0)
    assert curves[0, 1] == pytest.approx(1.0)
    assert curves[-1, 0] == pytest.approx(1.1 ** 2)
    assert curves[-1, 1] == pytest.approx(0.9 ** 2)


def test_cumulative_curves_skip_overlapping_windows():
    n = 20
    group_returns = np.zeros((n, 1))
    group_returns[:, 0] = 0.01
    rows, curves = cumulative_curves(group_returns, horizon=5)
    assert len(rows) == 4
    np.testing.assert_array_equal(rows, [0, 5, 10, 15])


# ---------------------------------------------------------------------
# turnover / autocorrelation
# ---------------------------------------------------------------------

def test_a_constant_factor_never_turns_over_and_is_perfectly_autocorrelated():
    market = _market()

    class _Constant(Factor):
        def compute_symbol(self, ctx, sym):
            # A distinct, deterministic constant per symbol -- distinct so no
            # two symbols tie for a bucket boundary, deterministic so the
            # test does not depend on Python's per-process hash randomization.
            return np.full(ctx.market.n_bars, float(SYMBOLS.index(sym)))

    panel = _panel_of(_Constant(), market)
    turnover = factor_turnover(panel, n_groups=3, rebalance=5)
    assert turnover['top'] == pytest.approx(0.0)
    assert turnover['bottom'] == pytest.approx(0.0)

    autocorr = factor_autocorr(panel, lags=(1, 5))
    for lag, value in autocorr.items():
        assert value == pytest.approx(1.0, abs=1e-6)


def test_a_reshuffled_factor_turns_over():
    market = _market(n_bars=200)

    class _Shuffled(Factor):
        def compute_symbol(self, ctx, sym):
            rng = np.random.default_rng(abs(hash((sym, ctx.market.n_bars))) % (2 ** 32))
            return rng.permutation(ctx.market.n_bars).astype('float64')

    panel = _panel_of(_Shuffled(), market)
    turnover = factor_turnover(panel, n_groups=len(SYMBOLS), rebalance=1)
    assert turnover['top'] > 0.3


# ---------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------

def test_coverage_summary_reports_the_warmup_head():
    market = _market()

    class _WarmsUp(Factor):
        def compute_symbol(self, ctx, sym):
            out = np.full(ctx.market.n_bars, 1.0)
            out[:30] = np.nan
            return out

    panel = _panel_of(_WarmsUp(), market)
    summary = coverage_summary(panel)
    assert summary['first_scored_bar'] == 30
    assert summary['n_symbols'] == len(SYMBOLS)
    assert summary['bars_with_full_coverage'] == market.n_bars - 30


# ---------------------------------------------------------------------
# annual slices (new: cross-year stability)
# ---------------------------------------------------------------------

def test_annual_slices_partitions_by_calendar_year():
    market = _market(n_bars=800)  # spans 2020 into 2022
    factor = _CheatFactor()
    panel = _panel_of(factor, market)
    fwd = panel_forward_returns(panel, market, [1])[1]
    slices = annual_slices(panel, fwd, market.dates, horizon=1, n_groups=3)
    assert set(slices) == {2020, 2021, 2022}
    assert sum(row['n_bars'] for row in slices.values()) == market.n_bars
    for row in slices.values():
        # the cheat factor is IC=1 within every year, not just in aggregate
        assert row['ic_mean'] == pytest.approx(1.0, abs=1e-6)
