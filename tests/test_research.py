"""Tests for the research package: bar-index-slicing/warmup correctness,
walk-forward splitting, and Optuna search-space resolution.

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

from research.objective import DEFAULT_SPARSE_PENALTY, _finite, score
from research.optimize import run_study
from research.runner_api import run_window
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


def _crashing_market(sym: str = 'AU', n_bars: int = 600, crash_bar: int = 400):
    """A monotone uptrend that gaps to near zero in a single bar.

    Monotone so the crossover is reliably long into the gap, and on a product
    with a large multiplier so one lot is enough to take equity through zero:
    that is what makes a run stop early and leaves its equity curve short at
    the *tail* rather than the head.
    """
    t = np.arange(n_bars, dtype='float64')
    close = 100.0 + 0.20 * t
    close[crash_bar:] = 1.0
    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    oi = np.full(n_bars, 5000.0)
    volume = np.full(n_bars, 1500.0)
    weighted = {'open': open_, 'high': high, 'low': low, 'close': close,
                'settle': close.copy(), 'oi': oi, 'volume': volume,
                'session': np.ones(n_bars)}
    contracts = {'C1': {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                        for i in range(n_bars)}}
    panel = build_panel(sym, n_bars, weighted=weighted, contracts=contracts,
                        contract_by_bar=['C1'] * n_bars, first_bar=0)
    return build_market({sym: panel}, n_bars)


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


def _m(sharpe, n_trades, **extra):
    return {'sharpe_ratio': sharpe, 'n_trades': n_trades, 'max_drawdown': 5.0,
            'n_forced_liquidations': 0, **extra}


def test_a_configuration_that_never_traded_is_not_the_best_one():
    """Under-trading used to be a *multiplicative* penalty, which scales a
    negative score toward zero -- i.e. improves it. A run that never traded
    therefore scored exactly 0.0 and outranked every honest loser, so a search
    over a space with no edge in it returned "do nothing" as `best_params`
    with a clean-looking value near zero instead of saying there was no edge.
    """
    wy = 1.0
    no_trades = score(_m(0.0, 0), window_years=wy)
    small_loss = score(_m(-0.2, 100), window_years=wy)
    assert no_trades < small_loss
    assert no_trades == pytest.approx(-DEFAULT_SPARSE_PENALTY)


@pytest.mark.parametrize('sharpe', [-3.0, -0.2, 0.0, 0.5, 2.0])
def test_trading_less_never_improves_a_score(sharpe):
    """The invariant the old multiplicative penalty broke. Scaling a score
    toward zero raises it whenever it is negative, so thinning the trade count
    used to be a way to *improve* a losing configuration."""
    dense = score(_m(sharpe, 100), window_years=1.0)
    for n in (50, 4, 2, 1, 0):
        assert score(_m(sharpe, n), window_years=1.0) <= dense
    assert score(_m(sharpe, 1), window_years=1.0) < dense


def test_sparse_penalty_dials_how_much_thin_evidence_counts():
    thin_but_good = _m(2.0, 1)
    lenient = score(thin_but_good, window_years=1.0, sparse_penalty=0.0)
    default = score(thin_but_good, window_years=1.0)
    strict = score(thin_but_good, window_years=1.0, sparse_penalty=1.0)
    assert lenient > default > strict
    # sparse_penalty=0 is pure shrinkage toward "no edge", never past it
    assert 0.0 < lenient < 2.0
    # a fully-sampled window is untouched by the dial
    dense = _m(1.5, 100)
    assert score(dense, window_years=1.0, sparse_penalty=0.0) == pytest.approx(
        score(dense, window_years=1.0, sparse_penalty=1.0))


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


def test_a_blown_up_trials_returns_are_not_shifted_forward_in_time(tmp_path, monkeypatch):
    """A run that ends early is missing bars at the *tail*; one whose
    indicators outran the pad is missing them at the *head*. Zero-filling the
    head unconditionally -- which is what this did -- slid every blown-up
    trial's whole return series forward in time. ``pbo_cscv`` slices that same
    axis into contiguous blocks and compares trials block by block, so a
    shifted trial was being scored against the wrong period in every split.
    """
    import research.optimize as optimize_mod
    from strategies.double_ma import DoubleMaStrategy

    seen = {}
    real_pbo = optimize_mod.pbo_cscv

    def spy(returns_matrix, **kwargs):
        seen['matrix'] = returns_matrix.copy()
        return real_pbo(returns_matrix, **kwargs)

    monkeypatch.setattr(optimize_mod, 'pbo_cscv', spy)

    # A trending series that collapses two thirds of the way through. The
    # crossover is long into it, and on this cash one lot is enough to take
    # equity through zero -- so trials stop mid-window and their curves end
    # short at the tail, which is the case that used to be mis-padded.
    out = run_study(
        strategy_cls=DoubleMaStrategy, symbols=['AU'], market=_crashing_market(),
        cash=30_000.0, n_trials=4, n_folds=2, probe_samples=3,
        results_dir=str(tmp_path),
    )
    assert out['report']['n_trials_blown_up'] > 0, 'expected a blow-up to exercise this path'

    matrix = seen['matrix']
    width = out['report']['holdout_window']['start'] - out['report']['reserve_bars']
    assert matrix.shape[1] == width

    spans = [(int(np.argmax(row != 0)), int(np.argmax(row[::-1] != 0)))
             for row in matrix if row.any()]
    # `reserve_bars` is sized to cover the whole space's warmup, so every trial
    # records from the window's first bar: no trial has anything to pad at the
    # head. A blown-up one pads at the tail instead -- which is exactly what
    # the old code put at the head, sliding the series forward by that much.
    assert max(lead for lead, _ in spans) <= 2, 'returns were shifted off the window start'
    assert any(trail > 10 for _, trail in spans), 'expected a truncated tail to pad'


def test_packaged_modules_do_not_import_top_level_scripts():
    """``pyproject.toml`` ships ``core``/``live``/``research``/... as packages
    and leaves ``runner.py`` and ``plotting.py`` at the repo root, uninstalled.
    ``live.signal`` and ``research.runner_api`` used to import
    ``run_single_backtest`` from ``runner``, so ``pip install .`` produced a
    tree that raised ``ModuleNotFoundError`` the first time either was used.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    packaged = ('core', 'datafeed', 'strategies', 'research', 'meta', 'live')
    top_level = {p.stem for p in root.glob('*.py')}
    assert {'runner', 'plotting'} <= top_level      # guard the premise

    offenders = []
    for package in packaged:
        for path in (root / package).rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [(node.module or '').split('.')[0]]
                elif isinstance(node, ast.Import):
                    names = [a.name.split('.')[0] for a in node.names]
                else:
                    continue
                offenders += [
                    f'{path.relative_to(root)}: {name}'
                    for name in names if name in top_level
                ]
    assert not offenders, f'packaged modules importing unpackaged scripts: {offenders}'
