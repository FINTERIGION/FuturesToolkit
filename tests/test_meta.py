"""Meta-labeling tests.

Uses the synthetic ``MarketData`` fixtures and the *tracked* ``double_ma``
strategy -- ``momentum_barrier`` is a private, gitignored module, so nothing
here may depend on it.

The load-bearing tests are the wrapper ones. A gate that silently blocks an
exit, or that shifts the baseline's warmup, produces a backtest that looks
fine and means nothing; those are the failures worth spending a test on.
"""

import datetime
import json
import logging

import numpy as np
import pandas as pd
import pytest

from core.engine import Engine
from meta.dataset import purged_train_mask, samples_from_trades, window_mask
from meta.features import DIRECTIONAL, FEATURE_NAMES, build_feature_arrays, feature_row
from meta.filter import make_meta_filtered, split_order, unfiltered
from meta.model import (
    PassThroughModel,
    WalkForwardMetaModel,
    fit_estimator,
    threshold_for,
)
from research.runner_api import run_window
from research.splits import Window
from research.warmup import probe_warmup
from runner import run_single_backtest
from strategies.base import SetupContext, Strategy
from strategies.double_ma import DoubleMaStrategy
from tests.conftest import build_market, build_oscillating_market, build_panel

N_BARS = 400


class _AlwaysReject:
    """Vetoes every gated order."""

    def proba(self, sym, i, x):
        return 0.0

    def threshold(self, i):
        return 0.5


class _AlwaysAllow:
    def proba(self, sym, i, x):
        return 1.0

    def threshold(self, i):
        return 0.5


def _trending_market(n_bars=N_BARS, symbols=('SA', 'CF')):
    """Two products on a sine-plus-drift path, so DoubleMa actually trades --
    this module's default bar count over the shared builder (tests/conftest.py)."""
    return build_oscillating_market(symbols=symbols, n_bars=n_bars)


def _run(market, cls, params=None, cash=200_000.0):
    return run_single_backtest(market, cls, params or {}, cash, 0.0)


# ----------------------------------------------------------------------
# split_order: the "never block an exit" rule, stated as a table
# ----------------------------------------------------------------------

@pytest.mark.parametrize('net,target,expected', [
    (0, 0, (0, 0)),        # nothing
    (0, 1, (0, 1)),        # fresh long: entirely gated
    (0, -1, (0, -1)),      # fresh short: entirely gated
    (2, 3, (2, 1)),        # add to a long: only the add is gated
    (2, 1, (1, 0)),        # trim: never gated
    (2, 0, (0, 0)),        # close: never gated
    (2, -2, (0, -2)),      # reversal: close is free, the new short is gated
    (-2, -3, (-2, -1)),    # add to a short
    (-2, -1, (-1, 0)),     # trim a short
    (-2, 2, (0, 2)),       # reversal from short
])
def test_split_order_only_gates_added_exposure(net, target, expected):
    assert split_order(net, target) == expected


# ----------------------------------------------------------------------
# The wrapper
# ----------------------------------------------------------------------

def test_passthrough_wrapper_reproduces_the_base_strategy_exactly():
    """The fair-baseline guarantee: wrapping with PassThrough changes nothing
    about the trades, only the warmup (which is the point of using it as the
    baseline arm)."""
    market = _trending_market()
    wrapped = _run(market, unfiltered(DoubleMaStrategy))

    # Same warmup floor, so the two runs cover the same bars and the trade
    # logs are comparable field by field.
    pad = probe_warmup(market, unfiltered(DoubleMaStrategy), {})
    bare = run_single_backtest(market, DoubleMaStrategy, {}, 200_000.0, 0.0, warmup_bars=pad)

    assert wrapped['result']['trade_logs'] == bare['result']['trade_logs']
    assert wrapped['metrics']['final_equity'] == bare['metrics']['final_equity']
    assert len(wrapped['result']['trade_logs']) > 0, 'fixture must actually trade'


def test_always_reject_produces_no_trades():
    market = _trending_market()
    out = _run(market, make_meta_filtered(DoubleMaStrategy, _AlwaysReject()))
    assert out['result']['trade_logs'] == []
    assert out['engine'].strategy.n_vetoed > 0


def test_always_allow_matches_passthrough():
    """A model that says yes to everything must be indistinguishable from no
    model at all -- otherwise the gate is changing orders it claims to pass."""
    market = _trending_market()
    allowed = _run(market, make_meta_filtered(DoubleMaStrategy, _AlwaysAllow()))
    base = _run(market, unfiltered(DoubleMaStrategy))
    assert allowed['result']['trade_logs'] == base['result']['trade_logs']


def test_gate_never_blocks_an_exit():
    """Open under a permissive model, then veto everything: the position must
    still be closed by the primary's own rule."""
    market = _trending_market()

    class _RejectAfterEntry:
        """Allows the first entry, then refuses everything."""

        def __init__(self):
            self.seen = 0

        def proba(self, sym, i, x):
            self.seen += 1
            return 1.0 if self.seen <= 1 else 0.0

        def threshold(self, i):
            return 0.5

    out = _run(market, make_meta_filtered(DoubleMaStrategy, _RejectAfterEntry()))
    logs = out['result']['trade_logs']
    assert len(logs) == 1, 'exactly one entry should have been allowed'
    # It closed on the strategy's own signal, not by running to the end of the
    # data -- `open_at_end` would mean the exit never fired.
    assert logs[0]['open_at_end'] == 0
    assert out['engine'].broker.net_position('SA') == 0


def test_reversal_veto_lands_flat_not_reversed():
    """net=+n, target=-n, vetoed -> 0. Not -n (the veto did nothing) and not
    +n (the veto also blocked the exit)."""
    market = _trending_market()

    class _Reversing(Strategy):
        params = {'flip_at': 250}

        def setup(self, ctx):
            for sym in ctx.symbols:
                ctx.add_indicator('c', sym, ctx.close(sym))

        def on_bar(self, ctx):
            if ctx.i == 200:
                ctx.set_target('SA', 2)
            elif ctx.i == self.p['flip_at']:
                ctx.set_target('SA', -2)

    filtered = make_meta_filtered(_Reversing, _RejectAfterOpen(_d(market, 250)))
    out = _run(market, filtered)
    assert out['engine'].broker.net_position('SA') == 0
    decisions = [t for t in out['result']['trade_logs']]
    assert len(decisions) == 1 and decisions[0]['direction'] == 'long'


class _RejectAfterOpen:
    """Allows everything before ``cutoff``, rejects from that date onward."""

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def proba(self, sym, when, x):
        return 0.0 if when >= self.cutoff else 1.0

    def threshold(self, when):
        return 0.5


def test_wrapper_preserves_params_and_space():
    wrapped = make_meta_filtered(DoubleMaStrategy, PassThroughModel())
    assert wrapped.params == DoubleMaStrategy.params
    assert wrapped.space == DoubleMaStrategy.space
    inst = wrapped(fast_period=3)
    assert inst.p['fast_period'] == 3
    assert inst.base.p['fast_period'] == 3, 'overrides must reach the wrapped strategy'


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------

def _setup_ctx(market):
    class _Null(Strategy):
        pass

    return SetupContext(Engine(market, _Null()))


def test_feature_arrays_are_leading_nan_only():
    """`SetupContext.add_indicator` runs `core.indicators.guard`, which rejects
    embedded NaN -- and TA-Lib downstream would go all-NaN on one. Degenerate
    bars (flat range, zero volume) are normal inputs, so this is a real risk,
    not a formality."""
    market = _trending_market()
    arrays = build_feature_arrays(_setup_ctx(market))

    assert set(n for n, _ in arrays) == set(FEATURE_NAMES)
    for (name, sym), arr in arrays.items():
        assert len(arr) == market.n_bars, f'{name}/{sym}'
        finite = np.isfinite(arr)
        assert finite.any(), f'{name}/{sym} is entirely invalid'
        first = int(np.argmax(finite))
        assert finite[first:].all(), f'{name}/{sym} has a gap after bar {first}'


def test_feature_arrays_survive_degenerate_bars():
    """Dark bars are flattened to open==high==low==close with zero volume by
    the data layer; the division guards have to hold."""
    n = 200
    close = np.linspace(100.0, 120.0, n)
    flat = np.zeros(n)
    code = 'SA509'
    panel = build_panel(
        'SA', n,
        weighted={'open': close, 'high': close, 'low': close, 'close': close,
                  'settle': close, 'oi': flat, 'volume': flat, 'session': np.ones(n)},
        contracts={code: {i: (close[i], close[i], close[i], close[i], close[i], 0.0, 0.0)
                          for i in range(n)}},
        contract_by_bar=[code] * n,
    )
    arrays = build_feature_arrays(_setup_ctx(build_market({'SA': panel}, n)))
    for (name, sym), arr in arrays.items():
        finite = np.isfinite(arr)
        first = int(np.argmax(finite)) if finite.any() else len(arr)
        assert finite[first:].all(), f'{name}/{sym} went non-finite on degenerate bars'


def test_directional_features_flip_with_side_and_neutral_ones_do_not():
    market = _trending_market()
    arrays = build_feature_arrays(_setup_ctx(market))
    i = market.n_bars - 1
    long_row = feature_row(arrays, 'SA', i, +1)
    short_row = feature_row(arrays, 'SA', i, -1)
    assert long_row is not None and short_row is not None

    for k, name in enumerate(FEATURE_NAMES):
        if name in DIRECTIONAL:
            assert long_row[k] == pytest.approx(-short_row[k]), name
        else:
            assert long_row[k] == pytest.approx(short_row[k]), name


def test_feature_row_matches_the_arrays_it_came_from():
    """The train/serve invariant: the gate and the offline sample builder call
    this same function, so if it ever disagreed with the arrays the two paths
    would silently diverge."""
    market = _trending_market()
    arrays = build_feature_arrays(_setup_ctx(market))
    i = market.n_bars - 5
    row = feature_row(arrays, 'CF', i, +1)
    for k, name in enumerate(FEATURE_NAMES):
        assert row[k] == pytest.approx(arrays[(name, 'CF')][i])


def test_feature_row_returns_none_inside_warmup():
    market = _trending_market()
    arrays = build_feature_arrays(_setup_ctx(market))
    assert feature_row(arrays, 'SA', 0, +1) is None


# ----------------------------------------------------------------------
# Dataset: alignment and purging
# ----------------------------------------------------------------------

def _samples_for(market):
    out = _run(market, unfiltered(DoubleMaStrategy))
    arrays = build_feature_arrays(_setup_ctx(market))
    return samples_from_trades(out['result']['trade_logs'], arrays), out


def test_samples_read_features_from_the_bar_before_the_fill():
    """`open_bar` is the fill bar; the decision was the close before it.
    Reading at `open_bar` would hand the model the bar it must predict into."""
    market = _trending_market()
    samples, _ = _samples_for(market)
    assert len(samples) > 0
    assert np.array_equal(samples.decision_bar, samples.open_bar - 1)
    assert (samples.decision_bar < samples.open_bar).all()


def test_samples_exclude_censored_trades():
    """A trade still open when the run ended has a censored label -- it is not
    a loss, its outcome is simply unknown -- so it must not become a sample."""
    market = _trending_market()
    samples, out = _samples_for(market)
    censored = {(t['symbol'], t['open_bar']) for t in out['result']['trade_logs']
                if t['open_at_end']}
    assert censored, 'fixture should end mid-position for this to test anything'
    present = set(zip(samples.symbol, samples.open_bar))
    assert not (present & censored)


def test_labels_match_the_sign_of_net_pnl():
    market = _trending_market()
    samples, out = _samples_for(market)
    by_bar = {(t['symbol'], t['open_bar']): t for t in out['result']['trade_logs']}
    for k in range(len(samples)):
        trade = by_bar[(samples.symbol[k], samples.open_bar[k])]
        assert samples.y[k] == int(trade['net_pnl'] > 0)


def test_purged_train_mask_excludes_trades_that_close_after_the_boundary():
    market = _trending_market()
    samples, _ = _samples_for(market)
    train = Window('train_1', 0, 250)
    valid = Window('valid_1', 260, 350)

    mask = purged_train_mask(samples, train, valid, embargo=10)
    assert mask.any()
    assert samples.close_bar[mask].max() < train.end
    # The purge is on close_bar, not open_bar: a trade opened before 250 but
    # still running at 250 must not be in the training set.
    straddling = (samples.open_bar < train.end) & (samples.close_bar >= train.end)
    assert not (mask & straddling).any()


def test_purge_keeps_trades_from_before_an_anchored_window_by_default():
    """`train_window.start` is `reserve_bars`, set by the slowest product's
    warmup -- a windowing choice, not a leakage rule. Trades that closed long
    before the cutoff are legitimate training data regardless of where the
    anchored window happens to begin, and on this universe one late listing
    would otherwise discard most of the sample."""
    market = _trending_market()
    samples, _ = _samples_for(market)
    train = Window('train_1', 200, 320)

    default = purged_train_mask(samples, train)
    anchored = purged_train_mask(samples, train, anchor_to_window=True)

    assert default.sum() > anchored.sum(), 'fixture must have trades before the anchor'
    assert samples.close_bar[default].max() < train.end, 'purge still holds'
    assert (samples.open_bar[anchored] >= train.start).all()


def test_purged_train_mask_raises_when_the_embargo_would_be_crossed():
    market = _trending_market()
    samples, _ = _samples_for(market)
    train = Window('train_1', 0, 250)
    valid = Window('valid_1', 251, 350)   # embargo far too small for the split
    with pytest.raises(AssertionError, match='purge failed'):
        purged_train_mask(samples, train, valid, embargo=100)


def test_window_mask_keys_on_the_decision_bar():
    market = _trending_market()
    samples, _ = _samples_for(market)
    w = Window('valid_1', 200, 300)
    mask = window_mask(samples, w)
    assert (samples.decision_bar[mask] >= 200).all()
    assert (samples.decision_bar[mask] < 300).all()


# ----------------------------------------------------------------------
# Walk-forward dispatch
# ----------------------------------------------------------------------

class _Const:
    """Estimator stub returning a fixed probability, tagged so the test can
    tell which fold answered."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1.0 - self.p, self.p], (len(X), 1))


def _d(market, i):
    return pd.Timestamp(market.dates[i]).date()


def test_walkforward_model_dispatches_on_date_and_is_silent_outside_its_windows():
    market = _trending_market()
    model = WalkForwardMetaModel([
        (_d(market, 100), _d(market, 199), _Const(0.10), 0.5),
        (_d(market, 300), _d(market, 399), _Const(0.90), 0.7),
    ])
    x = np.zeros(len(FEATURE_NAMES))

    assert model.proba('SA', _d(market, 150), x) == pytest.approx(0.10)
    assert model.threshold(_d(market, 150)) == pytest.approx(0.5)
    assert model.proba('SA', _d(market, 350), x) == pytest.approx(0.90)
    assert model.threshold(_d(market, 350)) == pytest.approx(0.7)
    # Both ends inclusive.
    assert model.proba('SA', _d(market, 100), x) is not None
    assert model.proba('SA', _d(market, 199), x) is not None

    # Before the first window, in the gap, and past the last: no opinion.
    for i in (0, 99, 200, 250, 299):
        assert model.proba('SA', _d(market, i), x) is None, i


def test_walkforward_gate_fires_inside_a_padded_research_window():
    """Regression: `run_window` evaluates a fold on a *slice* that starts `pad`
    bars early, so the engine's `ctx.i` is offset from the full market's bar
    numbering. A model addressed by bar index matched nothing there and let
    every order through, making the filtered arm score identically to the
    baseline at every threshold -- a silent no-op that looked like "the filter
    changes nothing". Addressing by date is what fixes it, and this is the
    test that distinguishes the two."""
    market = _trending_market()
    window = Window('valid_1', 300, N_BARS)
    pad = probe_warmup(market, unfiltered(DoubleMaStrategy), {})

    # Bounds expressed the way `walkforward` expresses them: the fold's own
    # window, in the FULL market's numbering. The slice the engine actually
    # sees starts at bar `window.start - pad`, so its local indices (0..159
    # here) never overlap 300..399 -- which is exactly why an index-addressed
    # model matched nothing and this test fails against one.
    reject_all = WalkForwardMetaModel([
        (_d(market, window.start), _d(market, window.end - 1), _Const(0.0), 0.5),
    ])
    filtered = run_window(market, make_meta_filtered(DoubleMaStrategy, reject_all), {},
                          window, cash=200_000.0, pad=pad)
    base = run_window(market, unfiltered(DoubleMaStrategy), {}, window,
                      cash=200_000.0, pad=pad)

    assert len(base['result']['trade_logs']) > 0, 'window must trade for this to test anything'
    assert filtered['result']['trade_logs'] == []
    assert filtered['engine'].strategy.n_vetoed > 0
    assert filtered['engine'].strategy.n_uncovered == 0


def test_uncovered_bars_pass_through_rather_than_being_vetoed():
    """Fail-open. If an uncovered bar were treated as a rejection, "the model
    had no coverage" and "the model said no" would be indistinguishable in the
    results."""
    import datetime

    market = _trending_market()
    far = datetime.date(2099, 1, 1)
    model = WalkForwardMetaModel([(far, far, _Const(0.0), 0.5)])  # covers nothing
    out = _run(market, make_meta_filtered(DoubleMaStrategy, model))
    base = _run(market, unfiltered(DoubleMaStrategy))
    assert out['result']['trade_logs'] == base['result']['trade_logs']
    assert out['engine'].strategy.n_uncovered > 0
    assert out['engine'].strategy.n_vetoed == 0


def test_threshold_for_keep_rate_one_is_an_exact_no_op():
    """keep_rate=1.0 is the self-check arm of the walk-forward, so it has to
    keep literally every trade, including the lowest-scoring one."""
    est = _Const(0.3)
    X = np.zeros((50, len(FEATURE_NAMES)))
    thr = threshold_for(est, X, 1.0)
    assert (est.predict_proba(X)[:, 1] >= thr).all()


def test_threshold_for_selects_the_requested_train_fraction():
    class _Ramp:
        def predict_proba(self, X):
            p = np.linspace(0.0, 1.0, len(X))
            return np.column_stack([1 - p, p])

    X = np.zeros((1000, len(FEATURE_NAMES)))
    thr = threshold_for(_Ramp(), X, 0.6)
    kept = (_Ramp().predict_proba(X)[:, 1] >= thr).mean()
    assert kept == pytest.approx(0.6, abs=0.01)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def _toy_fit(kind='lr'):
    """A tiny two-class training set. These tests are about persistence, not
    about the data, and the synthetic market's DoubleMa trades happen to be
    all winners."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype('int8')
    return fit_estimator(X, y, np.ones(len(y)), kind=kind), X


def test_save_load_round_trip_and_sidecar(tmp_path):
    from meta.model import FittedMetaModel, load_model, save_model

    est, X = _toy_fit()
    model = FittedMetaModel(estimator=est, threshold_value=0.42, keep_rate=0.6,
                            kind='lr', trained_through='2026-09-01')
    path = str(tmp_path / 'm.joblib')
    save_model(model, path, meta={'strategy': 'double_ma', 'params': {}})

    back = load_model(path)
    assert back.threshold_value == pytest.approx(0.42)
    assert back.keep_rate == pytest.approx(0.6)
    assert list(back.feature_names) == list(FEATURE_NAMES)

    with open(path + '.json', encoding='utf-8') as f:
        sidecar = json.load(f)
    assert sidecar['feature_names'] == list(FEATURE_NAMES)
    assert sidecar['strategy'] == 'double_ma'
    assert 'sklearn_version' in sidecar

    when = datetime.date(2026, 9, 1)
    assert back.proba('SA', when, X[0]) == pytest.approx(model.proba('SA', when, X[0]))


def test_load_refuses_a_model_whose_feature_contract_drifted(tmp_path):
    """A renamed or reordered feature does not raise on its own -- the
    estimator scores a vector of the right width and returns confident
    nonsense. Load time is the only cheap place to catch it."""
    from meta.model import FittedMetaModel, load_model, save_model

    est, _ = _toy_fit()
    model = FittedMetaModel(estimator=est, threshold_value=0.5, keep_rate=1.0, kind='lr')
    model.feature_names = ['something', 'entirely', 'different']
    path = str(tmp_path / 'stale.joblib')
    save_model(model, path, meta={})

    with pytest.raises(ValueError, match='different feature set'):
        load_model(path)


def test_fit_estimator_rejects_a_single_class():
    X = np.zeros((100, len(FEATURE_NAMES)))
    y = np.ones(100, dtype='int8')
    with pytest.raises(ValueError, match='both classes'):
        fit_estimator(X, y, np.ones(100), kind='lr')


# ----------------------------------------------------------------------
# Holdout guards (they run before any data is touched)
# ----------------------------------------------------------------------

def _report(tmp_path, **over):
    path = tmp_path / 'wf.json'
    body = {'verdict': 'SIGNAL: AUC 0.610 ...', 'strategy_lookup': 'double_ma'}
    body.update(over)
    path.write_text(json.dumps(body), encoding='utf-8')
    return str(path)


def _holdout_args(report, **over):
    import argparse

    ns = argparse.Namespace(report=report, keep_rate=0.6, kind='rf',
                            strategy='double_ma', force=False)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_holdout_refuses_a_no_edge_report(tmp_path):
    """Confirming a filter the walk-forward already rejected is asking a second
    judge for a different answer, and it spends a one-shot window to do it."""
    import meta_runner

    with pytest.raises(SystemExit):
        meta_runner.cmd_holdout(_holdout_args(_report(tmp_path, verdict='NO EDGE: ...')))


def test_holdout_refuses_a_second_run(tmp_path):
    import meta_runner

    report = _report(tmp_path, holdout_evaluated='2026-09-01T00:00:00')
    with pytest.raises(SystemExit):
        meta_runner.cmd_holdout(_holdout_args(report))


def test_holdout_force_overrides_both_guards(tmp_path, monkeypatch):
    """--force must get past the guards; it should then fail for a *data*
    reason, not a guard one."""
    import meta_runner

    report = _report(tmp_path, verdict='NO EDGE: ...',
                     holdout_evaluated='2026-09-01T00:00:00')
    called = {}

    def _boom(*a, **k):
        called['reached'] = True
        raise RuntimeError('stopped past the guards')

    monkeypatch.setattr(meta_runner, 'load_market', _boom)
    with pytest.raises((RuntimeError, KeyError)):
        meta_runner.cmd_holdout(_holdout_args(report, force=True))


# ----------------------------------------------------------------------
# Live signal
# ----------------------------------------------------------------------

def test_pending_after_the_last_bar_is_tomorrows_order():
    """The live-signal mechanism: the final bar's SIGNAL phase folds orders
    into `pending`, and no bar N+1 exists to fill them."""
    market = _trending_market()

    class _BuyOnLastBar(Strategy):
        def setup(self, ctx):
            for sym in ctx.symbols:
                ctx.add_indicator('c', sym, ctx.close(sym))

        def on_bar(self, ctx):
            if ctx.i == len(ctx._engine.market.dates) - 1:
                ctx.set_target('SA', 3)

    out = _run(market, _BuyOnLastBar)
    engine = out['engine']
    assert engine.pending.get('SA') == 3
    assert engine.broker.net_position('SA') == 0
    # target = current + pending, which is what `meta_runner.py signal` prints
    assert engine.broker.net_position('SA') + engine.pending.get('SA', 0) == 3


def test_last_decisions_is_per_bar_not_cumulative():
    market = _trending_market()
    out = _run(market, make_meta_filtered(DoubleMaStrategy, _AlwaysReject()))
    decisions = out['engine'].strategy.last_decisions
    assert set(decisions).issubset({'SA', 'CF'})
    for d in decisions.values():
        assert set(d) >= {'net', 'target', 'floor', 'side', 'allowed', 'proba', 'threshold'}


def test_load_warns_when_the_model_was_fitted_under_another_sklearn(tmp_path, caplog):
    """`save_model` has always recorded the fitting version in the sidecar and
    `load_model` never read it. scikit-learn does not support unpickling an
    estimator across versions -- it can load and predict differently rather
    than fail -- and a model file outlives the virtualenv that made it.
    """
    from meta.model import FittedMetaModel, load_model, save_model

    est, _ = _toy_fit()
    model = FittedMetaModel(estimator=est, threshold_value=0.5, keep_rate=1.0, kind='lr')
    path = str(tmp_path / 'old.joblib')
    save_model(model, path, meta={})

    with open(path + '.json', encoding='utf-8') as f:
        sidecar = json.load(f)
    sidecar['sklearn_version'] = '0.1.ancient'
    with open(path + '.json', 'w', encoding='utf-8') as f:
        json.dump(sidecar, f)

    with caplog.at_level(logging.WARNING, logger='meta.model'):
        back = load_model(path)                      # a warning, not a refusal
    assert back.threshold_value == pytest.approx(0.5)
    assert '0.1.ancient' in caplog.text


def test_load_refuses_a_model_and_sidecar_that_are_not_a_pair(tmp_path):
    """The two files are written together and copied around separately, so one
    can be swapped without the other. An integrity check, not a security one:
    whoever can write the .joblib can write the .json beside it."""
    from meta.model import FittedMetaModel, load_model, save_model

    est, _ = _toy_fit()
    model = FittedMetaModel(estimator=est, threshold_value=0.5, keep_rate=1.0, kind='lr')
    path = str(tmp_path / 'mismatched.joblib')
    save_model(model, path, meta={})

    with open(path + '.json', encoding='utf-8') as f:
        sidecar = json.load(f)
    sidecar['feature_names'] = list(FEATURE_NAMES)[::-1]
    with open(path + '.json', 'w', encoding='utf-8') as f:
        json.dump(sidecar, f)

    with pytest.raises(ValueError, match='not the pair'):
        load_model(path)


def test_load_still_works_without_a_sidecar(tmp_path):
    """Artifacts predating the sidecar are still usable; the model is
    self-describing enough on its own."""
    import os

    from meta.model import FittedMetaModel, load_model, save_model

    est, _ = _toy_fit()
    model = FittedMetaModel(estimator=est, threshold_value=0.5, keep_rate=1.0, kind='lr')
    path = str(tmp_path / 'bare.joblib')
    save_model(model, path, meta={})
    os.remove(path + '.json')

    assert load_model(path).threshold_value == pytest.approx(0.5)
