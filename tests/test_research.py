"""Tests for the research package: bar-index-slicing/warmup correctness,
walk-forward splitting, Optuna search-space resolution, and meta-labeling's
purge/gating logic.

Uses synthetic MarketData throughout (via tests/conftest.py's builders, plus
a locally-built trending/cyclical series so the bundled strategies actually
generate trades) -- no dependency on data/*.csv, matching the rest of the
suite. Tests that must hold for *any* strategy are parametrized over
``strategies.discover_strategies()`` rather than naming one strategy, so a
new strategy dropped into strategies/ is covered automatically.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.engine import Engine
from core.market import slice_market
from core.params import Int
from strategies import discover_strategies
from strategies.base import BarContext, SetupContext, Strategy

from research.gating import MetaFilteredStrategy
from research.metalabel import (
    _slice_metrics, evaluate_meta_backtest, extract_events, fit_meta_model, purge_events,
)
from research.objective import _finite, score
from research.optimize import run_study
from research.runner_api import run_window, slice_start
from research.space import resolve_space
from research.splits import Window, anchored_walk_forward
from research.warmup import probe_warmup

from tests.conftest import build_market, build_panel

ALL_STRATEGIES = discover_strategies()
TUNABLE_STRATEGIES = {
    name: cls for name, cls in ALL_STRATEGIES.items()
    if getattr(cls, 'params', None)
}


def _trending_market(n_bars: int = 900, seed: int = 0):
    """A single-product synthetic market with a trend + a ~15-bar cycle, so
    both a crossover and a mean-reversion strategy generate real trades."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars, dtype='float64')
    close = 100.0 + 0.03 * t + 4.0 * np.sin(t / 15.0) + rng.normal(0, 0.3, n_bars)
    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.5, n_bars)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, n_bars)
    volume = rng.uniform(1000, 2000, n_bars)
    oi = rng.uniform(5000, 6000, n_bars)

    weighted = {
        'open': open_, 'high': high, 'low': low, 'close': close, 'settle': close.copy(),
        'oi': oi, 'volume': volume, 'session': np.ones(n_bars),
    }
    contracts = {
        'C1': {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i]) for i in range(n_bars)},
    }
    panel = build_panel('SA', n_bars, weighted=weighted, contracts=contracts,
                         contract_by_bar=['C1'] * n_bars, first_bar=0)
    return build_market({'SA': panel}, n_bars)


@pytest.fixture(scope='module')
def market():
    return _trending_market()


# ---------------------------------------------------------------------
# core.market.slice_market / core.engine warmup_bars
# ---------------------------------------------------------------------

def test_slice_market_rebases_bar_indices(market):
    sliced = slice_market(market, 100, 300)
    assert sliced.n_bars == 200
    panel = market.products['SA']
    sliced_panel = sliced.products['SA']
    assert np.array_equal(sliced_panel.weighted['close'], panel.weighted['close'][100:300])
    # A contract row at absolute bar 150 should show up at local bar 50.
    assert np.array_equal(sliced_panel.contracts['C1'].row_at(50), panel.contracts['C1'].row_at(150))
    assert sliced_panel.first_bar == 0


def test_slice_market_shifts_first_bar(market):
    # first_bar starts at 0; slicing from before it keeps it at the shifted position.
    sliced = slice_market(market, 50, 300)
    assert sliced.products['SA'].first_bar == 0  # max(0, 0-50) == 0, listing predates the slice


def test_equity_curve_starts_after_indicator_warmup_not_the_pad(market):
    """A pad too small for the indicators must not leave the un-tradeable
    bars in the equity curve: `on_bar` never fires there, so they would be
    flat zero-return bars no decision produced, diluting the metrics."""
    class _LateIndicator(Strategy):
        def setup(self, ctx):
            arr = np.ones(len(ctx.dates))
            arr[:80] = np.nan          # first valid bar is 80, well past the pad
            ctx.add_indicator('late', 'SA', arr)

        def on_bar(self, ctx):
            pass

    strat = _LateIndicator()
    engine = Engine(market, strat, initial_cash=100_000.0, warmup_bars=50)
    strat.setup(SetupContext(engine))
    result = engine.run_backtest(SetupContext, BarContext)

    assert engine.warmup_index == 80
    assert engine.record_start == 80          # not the supplied 50
    assert len(result['equity_records']) == market.n_bars - 80


def test_record_start_never_precedes_the_caller_supplied_floor(market):
    """`add_indicator` raising `warmup_index` moves the curve's start out;
    indicators valid from bar 0 leave the caller's own floor standing."""
    class _FixedIndicator(Strategy):
        def setup(self, ctx):
            ctx.add_indicator('const', 'SA', np.ones(len(ctx.dates)))

        def on_bar(self, ctx):
            pass

    strat = _FixedIndicator()
    engine = Engine(market, strat, initial_cash=100_000.0, warmup_bars=50)
    strat.setup(SetupContext(engine))
    assert engine.record_start == 50


def test_run_window_effective_start_is_the_first_recorded_bar(market):
    """`effective_start` is what callers index `equity_records` against, so
    it has to stay the absolute bar of entry 0 even when the pad falls short."""
    class _LateIndicator(Strategy):
        params = {}

        def setup(self, ctx):
            arr = np.ones(len(ctx.dates))
            arr[:120] = np.nan
            ctx.add_indicator('late', 'SA', arr)

        def on_bar(self, ctx):
            pass

    window = Window('w', 300, 600)
    out = run_window(market, _LateIndicator, {}, window, cash=100_000.0, pad=50)
    # Slice starts at bar 250; the indicator needs 120 slice-local bars.
    assert out['effective_start'] == 250 + 120
    assert len(out['result']['equity_records']) == window.end - out['effective_start']


def test_engine_warmup_bars_excludes_pad_from_equity_curve(market):
    class _FixedIndicator(Strategy):
        def setup(self, ctx):
            ctx.add_indicator('const', 'SA', np.ones(len(ctx.dates)))

        def on_bar(self, ctx):
            pass

    strat = _FixedIndicator()
    engine = Engine(market, strat, initial_cash=100_000.0, warmup_bars=50)
    strat.setup(SetupContext(engine))
    result = engine.run_backtest(SetupContext, BarContext)
    assert len(result['equity_records']) == market.n_bars - 50
    assert result['equity_records'][0]['daily_return'] == 0.0  # first recorded bar vs. unchanged cash


# ---------------------------------------------------------------------
# Leakage sentinel: a window's backtest must be identical whether or not
# the parent market contains data past the window's end.
# ---------------------------------------------------------------------

def test_windowed_backtest_does_not_see_future_bars(market):
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    cutoff = 500
    window = Window('w', 300, cutoff)

    truncated = slice_market(market, 0, cutoff)  # parent literally has no bars past `cutoff`
    out_truncated = run_window(truncated, DoubleMaStrategy, params, window, cash=100_000.0, pad=50)

    shocked = _trending_market(n_bars=market.n_bars, seed=0)
    shocked.products['SA'].weighted['close'][cutoff:] *= 100.0  # huge shock, only after the window
    shocked.products['SA'].weighted['high'][cutoff:] *= 100.0
    shocked.products['SA'].weighted['low'][cutoff:] *= 100.0
    shocked.products['SA'].weighted['open'][cutoff:] *= 100.0
    shocked.products['SA'].weighted['settle'][cutoff:] *= 100.0
    out_shocked = run_window(shocked, DoubleMaStrategy, params, window, cash=100_000.0, pad=50)

    assert out_truncated['result']['trade_logs'] == out_shocked['result']['trade_logs']
    assert [r['equity'] for r in out_truncated['result']['equity_records']] == \
           [r['equity'] for r in out_shocked['result']['equity_records']]


def test_run_window_bar_indices_are_absolute_not_slice_local(market):
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 5, 'slow_period': 20, 'lots': 1}
    window = Window('mid', 400, 800)
    out = run_window(market, DoubleMaStrategy, params, window, cash=100_000.0, pad=50)
    assert out['lo'] == 350  # window.start(400) - pad(50)

    dates = market.dates
    for t in out['result']['trade_logs']:
        assert window.start <= t['open_bar'] < window.end
        assert dates[t['open_bar']] == np.datetime64(t['open_date'])
        if t['close_bar'] is not None:
            assert dates[t['close_bar']] == np.datetime64(t['close_date'])


# ---------------------------------------------------------------------
# Anchored walk-forward invariants
# ---------------------------------------------------------------------

@pytest.mark.parametrize('n_bars,reserve,n_folds,embargo', [
    (2000, 50, 4, 10), (900, 0, 1, 0), (1500, 300, 6, 20), (500, 100, 2, 5),
])
def test_anchored_walk_forward_invariants(n_bars, reserve, n_folds, embargo):
    folds, holdout = anchored_walk_forward(n_bars, reserve_bars=reserve, n_folds=n_folds, embargo=embargo)
    assert holdout.end == n_bars
    prev_end = reserve
    for train, valid in folds:
        assert train.start == reserve
        assert train.end > train.start
        assert train.end >= prev_end
        assert valid.start == train.end + embargo
        assert valid.start < valid.end
        assert valid.end <= holdout.start
        prev_end = train.end
    assert folds[-1][1].end == holdout.start


def test_anchored_walk_forward_rejects_infeasible_split():
    with pytest.raises(ValueError):
        anchored_walk_forward(200, reserve_bars=50, n_folds=10, embargo=20)


# ---------------------------------------------------------------------
# Search-space resolution (parametrized over every tunable strategy)
# ---------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(TUNABLE_STRATEGIES))
def test_resolve_space_covers_tunable_params(name):
    cls = TUNABLE_STRATEGIES[name]
    space = resolve_space(cls)
    fixed = set(getattr(cls, 'fixed_params', ()) or ())
    assert space  # every bundled tunable strategy should yield a non-empty space
    for key in space:
        assert key in cls.params
        assert key not in fixed


def test_resolve_space_infers_when_undeclared():
    class _Undeclared(Strategy):
        params = {'period': 20, 'threshold': 0.5}

    space = resolve_space(_Undeclared)
    assert set(space) == {'period', 'threshold'}


def test_resolve_space_rejects_strategy_with_nothing_to_tune():
    class _Empty(Strategy):
        params = {}

    with pytest.raises(ValueError):
        resolve_space(_Empty)


def test_resolve_space_cli_override_takes_priority():
    class _S(Strategy):
        params = {'period': 20}
        space = {'period': Int(5, 50)}

    override = {'period': Int(1, 3)}
    space = resolve_space(_S, override)
    assert space['period'] == Int(1, 3)


# ---------------------------------------------------------------------
# probe_warmup
# ---------------------------------------------------------------------

def test_probe_warmup_matches_talib_lookback(market):
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 5, 'slow_period': 20, 'lots': 1}
    warmup = probe_warmup(market, DoubleMaStrategy, params)
    assert warmup == 19  # SMA(20) needs 20 samples -> first valid index 19


# ---------------------------------------------------------------------
# objective scoring
# ---------------------------------------------------------------------

def test_finite_clamps_inf_and_nan():
    assert _finite(float('inf')) == 0.0
    assert _finite(float('-inf')) == 0.0
    assert _finite(float('nan')) == 0.0
    assert _finite(1.25) == 1.25


def test_score_on_empty_metrics_is_worst_case():
    assert score({}, window_years=1.0) == -10.0


def test_score_penalizes_excess_drawdown():
    base = {'sharpe_ratio': 1.0, 'n_trades': 100, 'max_drawdown': 10.0, 'n_forced_liquidations': 0}
    high_dd = {**base, 'max_drawdown': 80.0}
    assert score(high_dd, window_years=1.0, dd_cap=0.35) < score(base, window_years=1.0, dd_cap=0.35)


# ---------------------------------------------------------------------
# Gating equivalence (parametrized over every discovered strategy)
# ---------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(ALL_STRATEGIES))
def test_meta_filtered_strategy_threshold_zero_matches_baseline(market, name):
    cls = ALL_STRATEGIES[name]
    params = dict(getattr(cls, 'params', {}) or {})
    window = Window('full', 0, market.n_bars)

    baseline = run_window(market, cls, params, window, cash=100_000.0, pad=0)
    proba = {'SA': np.full(market.n_bars, 0.5)}  # always-valid, never NaN

    def factory(**_ignored):
        return MetaFilteredStrategy(cls(**params), proba, meta_threshold=0.0)

    gated = run_window(market, factory, {}, window, cash=100_000.0, pad=0)
    assert baseline['result']['trade_logs'] == gated['result']['trade_logs']


@pytest.mark.parametrize('name', sorted(TUNABLE_STRATEGIES))
def test_meta_filtered_strategy_threshold_above_one_blocks_all_entries(market, name):
    cls = TUNABLE_STRATEGIES[name]
    params = dict(getattr(cls, 'params', {}) or {})
    window = Window('full', 0, market.n_bars)
    proba = {'SA': np.full(market.n_bars, 0.5)}

    # A cross-sectional strategy needs more than one product to define a
    # cross-section at all, so it never opens a position on this
    # single-product ``market`` fixture -- nothing for the gate to reject.
    baseline = run_window(market, cls, params, window, cash=100_000.0, pad=0)
    if not baseline['result']['trade_logs']:
        pytest.skip(f'{name} never opens a position on this single-product market')

    def factory(**_ignored):
        return MetaFilteredStrategy(cls(**params), proba, meta_threshold=1.1)

    gated = run_window(market, factory, {}, window, cash=100_000.0, pad=0)
    assert gated['result']['trade_logs'] == []
    assert gated['engine'].strategy.threshold_rejected_count > 0


def test_gating_reads_proba_at_absolute_bars_on_an_offset_slice(market):
    """A window whose pad does not reach bar 0 makes the engine's bar 0 land
    at an absolute bar > 0. ``proba`` is indexed absolutely, so the gate must
    apply ``bar_offset``; without it every lookup lands ``lo`` bars early and
    a fully-permissive array reads as all-NaN (i.e. blocks everything)."""
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    window, pad = Window('offset', 400, 700), 50
    lo = slice_start(window, pad)
    assert lo == 350, 'precondition: this window/pad really must be offset from bar 0'

    baseline = run_window(market, DoubleMaStrategy, params, window, cash=100_000.0, pad=pad)

    # Permissive over exactly the window's absolute bars, NaN everywhere else.
    proba = {'SA': np.full(market.n_bars, np.nan)}
    proba['SA'][window.start:window.end] = 1.0

    def factory(**_ignored):
        return MetaFilteredStrategy(
            DoubleMaStrategy(**params), proba, bar_offset=lo, meta_threshold=0.5,
        )

    gated = run_window(market, factory, {}, window, cash=100_000.0, pad=pad)
    assert baseline['result']['trade_logs'] == gated['result']['trade_logs']
    assert gated['engine'].strategy.nan_count == 0

    def wrong_offset_factory(**_ignored):
        return MetaFilteredStrategy(
            DoubleMaStrategy(**params), proba, bar_offset=0, meta_threshold=0.5,
        )

    mis = run_window(market, wrong_offset_factory, {}, window, cash=100_000.0, pad=pad)
    assert mis['result']['trade_logs'] == [], 'slice-local lookup should read the NaN region'


def test_gating_treats_out_of_range_bars_as_blocked(market):
    """A proba array shorter than the run must block rather than IndexError."""
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    window = Window('full', 0, market.n_bars)
    # Length 1: only bar 0 is in range, and bar 0 is always inside indicator
    # warmup, so every lookup the gate actually performs is out of range.
    proba = {'SA': np.full(1, 1.0)}

    def factory(**_ignored):
        return MetaFilteredStrategy(DoubleMaStrategy(**params), proba, meta_threshold=0.5)

    gated = run_window(market, factory, {}, window, cash=100_000.0, pad=0)
    assert gated['result']['trade_logs'] == []
    assert gated['engine'].strategy.out_of_range_count > 0

    # ... unless the caller says missing data should fall the other way.
    def passing_factory(**_ignored):
        return MetaFilteredStrategy(
            DoubleMaStrategy(**params), proba, meta_threshold=0.5, on_missing='pass')

    passed = run_window(market, passing_factory, {}, window, cash=100_000.0, pad=0)
    baseline = run_window(market, DoubleMaStrategy, params, window, cash=100_000.0, pad=0)
    assert passed['result']['trade_logs'] == baseline['result']['trade_logs']
    assert passed['engine'].strategy.blocked_count == 0


# ---------------------------------------------------------------------
# _slice_metrics / evaluate_meta_backtest
# ---------------------------------------------------------------------

def test_slice_metrics_attributes_trades_by_close_bar(market):
    """Trades belong to the window their P&L is booked in, and the equity
    slice is taken relative to the *run's* start, not to bar 0."""
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    full = Window('full', 100, 800)
    run = run_window(market, DoubleMaStrategy, params, full, cash=100_000.0, pad=100)
    result = run['result']

    inner = Window('inner', 300, 600)
    metrics = _slice_metrics(run, inner, 100_000.0)

    expected = [
        t for t in result['trade_logs']
        if t['close_bar'] is not None and inner.start <= t['close_bar'] < inner.end
    ]
    assert metrics.get('n_trades', 0) == len(expected)
    assert expected, 'fixture should produce trades closing inside the inner window'
    # Equity slice is offset by the run's own start, not read from index 0.
    assert len(result['equity_records'][inner.start - full.start:inner.end - full.start]) == inner.n_bars


def test_evaluate_meta_backtest_isolates_the_filter_from_oof_coverage(market):
    """gated-vs-baseline must reflect the filter's decisions and nothing else.

    Most bars carry no out-of-fold prediction (only events inside a fold's
    valid window get one), so if those were blocked the gated arm would differ
    from the baseline for reasons that have nothing to do with the model. The
    load-bearing assertion is the threshold=0.0 case: when the filter rejects
    nothing, gated must be trade-for-trade identical to the baseline.
    """
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    bundle = fit_meta_model(
        market, DoubleMaStrategy, params, cash=100_000.0,
        n_folds=3, embargo=5, holdout_frac=0.2, min_events_per_fold=5, reserve_bars=20,
    )

    permissive = evaluate_meta_backtest({**bundle, 'threshold': 0.0}, market, n_random=3, seed=0)
    assert permissive['n_rejected'] == 0
    assert permissive['gated_metrics'] == permissive['baseline_metrics'], (
        'with nothing rejected the gated arm must reproduce the baseline exactly'
    )
    assert permissive['gate_counters']['blocked_total'] == 0
    # Nothing rejected -> the random comparison is a point mass, not a verdict.
    assert permissive['random_baseline']['degenerate'] is True
    assert permissive['random_baseline']['beats_random'] is None
    assert permissive['random_baseline']['gated_percentile'] is None

    strict = evaluate_meta_backtest({**bundle, 'threshold': 1.1}, market, n_random=3, seed=0)
    assert strict['n_kept'] == 0
    assert strict['n_rejected'] == strict['n_events_scored']
    assert strict['gate_counters']['threshold_rejected'] > 0
    assert strict['gated_metrics'] != strict['baseline_metrics']


def test_evaluate_meta_backtest_reports_consistent_counts(market):
    """Bookkeeping invariants for the gated-vs-random comparison."""
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    bundle = fit_meta_model(
        market, DoubleMaStrategy, params, cash=100_000.0,
        n_folds=3, embargo=5, holdout_frac=0.2, min_events_per_fold=5, reserve_bars=20,
    )
    res = evaluate_meta_backtest(bundle, market, n_random=5, seed=0)

    assert res['n_kept'] + res['n_rejected'] == res['n_events_scored']
    assert res['window']['start'] < res['window']['end']
    assert res['rejected_net_pnl_positive'] == (res['rejected_net_pnl'] > 0)
    for key in ('baseline_metrics', 'gated_metrics', 'random_baseline'):
        assert res[key], f'{key} should not be empty'
    assert res['random_baseline']['n_random'] == 5

    # Coverage is a fraction of the signals actually present in the window.
    assert res['n_events_scored'] <= res['n_events_in_window']
    assert 0.0 < res['oof_coverage'] <= 1.0

    # The evaluation window must stay clear of the holdout the bundle locked away.
    _folds, holdout = anchored_walk_forward(
        market.n_bars, reserve_bars=bundle['reserve_bars'], n_folds=bundle['n_folds'],
        embargo=bundle['embargo'], holdout_frac=bundle['holdout_frac'],
    )
    assert res['window']['end'] <= holdout.start


# ---------------------------------------------------------------------
# runner.py param plumbing (the optimize -> full-backtest handoff)
# ---------------------------------------------------------------------

def test_resolve_params_precedence(tmp_path):
    """--param beats --lots beats --params-from beats the class defaults."""
    import argparse
    import json as _json

    from runner import parse_param_value, resolve_params
    from strategies.double_ma import DoubleMaStrategy

    report = tmp_path / 'best.json'
    report.write_text(_json.dumps(
        {'best_params': {'fast_period': 9, 'slow_period': 44, 'lots': 2}}), encoding='utf-8')

    args = argparse.Namespace(params_from=str(report), lots=None, param=None)
    assert resolve_params(DoubleMaStrategy, args) == {'fast_period': 9, 'slow_period': 44, 'lots': 2}

    args = argparse.Namespace(params_from=str(report), lots=7, param=['fast_period=3'])
    assert resolve_params(DoubleMaStrategy, args) == {'fast_period': 3, 'slow_period': 44, 'lots': 7}

    # No flags at all -> no overrides, so the class defaults stand untouched.
    args = argparse.Namespace(params_from=None, lots=None, param=None)
    assert resolve_params(DoubleMaStrategy, args) == {}

    assert parse_param_value('n=5') == ('n', 5)
    assert parse_param_value('r=0.5') == ('r', 0.5)
    assert parse_param_value('flag=true') == ('flag', True)
    assert parse_param_value('mode=trend') == ('mode', 'trend')
    with pytest.raises(ValueError):
        parse_param_value('nope')


# ---------------------------------------------------------------------
# extract_events / purge_events
# ---------------------------------------------------------------------

def test_extract_events_matches_ledger_trade_count(market):
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    window = Window('full', 0, market.n_bars)
    ext = extract_events(market, DoubleMaStrategy, params, window, cash=100_000.0, pad=0)
    assert len(ext['events']) == len(ext['run']['result']['trade_logs'])
    assert len(ext['events']) > 10  # the cyclical series should produce plenty of crossovers
    for e in ext['events']:
        assert e['signal_bar'] < e['open_bar']
        assert e['weight'] == abs(e['net_pnl'])
        assert e['label'] == (e['net_pnl'] > 0)


def test_purge_events_removes_boundary_crossing_and_embargoed_events():
    events = [
        {'symbol': 'SA', 'signal_bar': 50, 'open_bar': 51, 'close_bar': 60, 'net_pnl': 10.0, 'label': True, 'weight': 10.0},   # fully inside train
        {'symbol': 'SA', 'signal_bar': 90, 'open_bar': 91, 'close_bar': 150, 'net_pnl': 5.0, 'label': True, 'weight': 5.0},    # opens in train, closes inside valid -> purge
        {'symbol': 'SA', 'signal_bar': 95, 'open_bar': 96, 'close_bar': 105, 'net_pnl': -3.0, 'label': False, 'weight': 3.0},  # holding interval reaches into embargo padding -> purge
        {'symbol': 'SA', 'signal_bar': 120, 'open_bar': 121, 'close_bar': 130, 'net_pnl': 7.0, 'label': True, 'weight': 7.0},  # inside valid window
    ]
    train = Window('train', 0, 100)
    valid = Window('valid', 110, 200)
    train_events, valid_events = purge_events(events, train, valid, embargo=10)

    assert valid_events == [events[3]]
    train_bars = {e['signal_bar'] for e in train_events}
    assert 50 in train_bars
    assert 90 not in train_bars   # holding interval [91,150] overlaps embargoed valid region
    assert 95 not in train_bars   # holding interval [96,105] overlaps [valid.start-embargo, ...) = [100, ...)
    for e in train_events:
        close_bar = e['close_bar']
        assert not (e['open_bar'] < valid.end + 10 and close_bar >= valid.start - 10)


# ---------------------------------------------------------------------
# End-to-end smoke tests (small, fast configurations)
# ---------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(TUNABLE_STRATEGIES))
def test_optimize_run_study_end_to_end(market, name, tmp_path):
    cls = TUNABLE_STRATEGIES[name]
    out = run_study(
        strategy_cls=cls, symbols=['SA'], market=market,
        n_trials=4, n_folds=2, probe_samples=4, results_dir=str(tmp_path),
    )
    report = out['report']
    space_keys = set(report['space'])
    assert set(report['best_params']) >= space_keys
    assert report['holdout_evaluated'] is False
    for key in ('is_oos_decay', 'pbo', 'dsr', 'plateau'):
        assert key in report['diagnostics']


def test_optimize_run_study_never_touches_holdout_bars(market, tmp_path, monkeypatch):
    from strategies.double_ma import DoubleMaStrategy
    import research.optimize as optimize_mod

    seen_windows = []
    real_run_window = optimize_mod.run_window

    def spy(market_arg, strategy_cls, params, window, **kwargs):
        seen_windows.append(window)
        return real_run_window(market_arg, strategy_cls, params, window, **kwargs)

    monkeypatch.setattr(optimize_mod, 'run_window', spy)

    out = run_study(
        strategy_cls=DoubleMaStrategy, symbols=['SA'], market=market,
        n_trials=3, n_folds=2, probe_samples=3, results_dir=str(tmp_path),
    )
    holdout_start = out['report']['holdout_window']['start']
    for w in seen_windows:
        assert w.end <= holdout_start, f"window {w} reaches into the holdout region"


def test_fit_meta_model_end_to_end(market):
    from strategies.double_ma import DoubleMaStrategy

    params = {'fast_period': 3, 'slow_period': 8, 'lots': 1}
    bundle = fit_meta_model(
        market, DoubleMaStrategy, params, cash=100_000.0,
        n_folds=3, embargo=5, holdout_frac=0.2, min_events_per_fold=5,
        reserve_bars=20,  # skip the 252-bar z-score-driven default; this market is only 900 bars
    )
    assert 0.0 <= bundle['threshold'] <= 1.0
    assert bundle['n_events_oof'] > 0
    assert bundle['features']
    # The deployed pipeline must have been fitted on exactly the feature set
    # recorded alongside it -- anything else silently mis-orders columns at
    # predict time.
    assert bundle['pipeline'].n_features_in_ == len(bundle['features'])
