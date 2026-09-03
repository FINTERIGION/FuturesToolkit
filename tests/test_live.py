"""Live-signal tests.

The load-bearing property is the one the whole module rests on: after the
last bar, ``target == simulated position + engine.pending``. Everything else
here guards the *shared* path -- a plain strategy and a meta-gated one must
come out of the same function with the same row shape, differing only in the
fields a gate can fill in.

Uses the synthetic fixtures and the tracked ``double_ma`` strategy;
``momentum_barrier`` and friends are private, gitignored modules that nothing
here may depend on.
"""

import json

import numpy as np
import pytest

from strategies.base import Strategy
from strategies.double_ma import DoubleMaStrategy
from tests.conftest import build_market, build_panel

from live.report import render, save
from live.signal import SignalSpec, compute_signal

N_BARS = 300


class _FakeModel:
    """Enough of ``FittedMetaModel`` for the gate and the report header."""

    keep_rate = 0.5
    threshold_value = 0.5
    trained_through = '2024-01-01'

    def __init__(self, p):
        self._p = p

    def proba(self, sym, when, x):
        return self._p

    def threshold(self, when):
        return 0.5


def _trending_market(n_bars=N_BARS, symbols=('SA', 'CF')):
    rng = np.random.default_rng(7)
    products = {}
    for k, sym in enumerate(symbols):
        t = np.arange(n_bars, dtype='float64')
        close = 100.0 + 20.0 * np.sin(t / (18.0 + 5 * k)) + 0.02 * t + rng.normal(0, 0.6, n_bars)
        close = np.maximum(close, 5.0)
        code = f'{sym}509'
        contracts = {code: {i: (close[i], close[i] + 1, close[i] - 1, close[i], close[i],
                                1000.0 + i, 500.0 + i) for i in range(n_bars)}}
        products[sym] = build_panel(
            sym, n_bars,
            weighted={
                'open': close, 'high': close + 1.0, 'low': close - 1.0,
                'close': close, 'settle': close,
                'oi': 1000.0 + np.arange(n_bars), 'volume': 500.0 + np.arange(n_bars),
                'session': np.ones(n_bars),
            },
            contracts=contracts,
            contract_by_bar=[code] * n_bars,
        )
    return build_market(products, n_bars)


def _spec(cls=DoubleMaStrategy, **kw):
    return SignalSpec(
        strategy_cls=cls, params=kw.pop('params', {}),
        symbols=list(kw.pop('symbols', ['SA', 'CF'])),
        cash=kw.pop('cash', 200_000.0), slippage=0.0, **kw,
    )


# ----------------------------------------------------------------------
# The mechanism
# ----------------------------------------------------------------------

def test_target_is_simulated_position_plus_pending():
    """The one identity the module exists to compute."""
    market = _trending_market()
    report = compute_signal(market, _spec())
    engine = report.engine
    for row in report.rows:
        net = engine.broker.net_position(row.symbol)
        assert row.current_simulated == net
        assert row.delta == engine.pending.get(row.symbol, 0)
        assert row.target == net + row.delta


def test_a_plain_strategy_needs_no_model():
    """The point of the split: `strategies/` works without meta-labeling."""
    report = compute_signal(_trending_market(), _spec())
    assert not report.is_meta
    assert {r.symbol for r in report.rows} == {'SA', 'CF'}
    assert all(r.proba is None and r.verdict is None for r in report.rows)
    # DoubleMa is always in the market once warm, so it has a live target.
    assert all(r.target != 0 for r in report.rows)


def test_rows_carry_the_execution_contract():
    report = compute_signal(_trending_market(), _spec())
    assert [r.contract for r in report.rows] == ['SA509', 'CF509']
    assert all(r.tradable for r in report.rows)


def test_symbols_absent_from_the_market_are_reported_flat_not_dropped():
    """A requested product that never loaded must still show up, marked
    untradable -- silently omitting it reads as 'nothing to do today'."""
    report = compute_signal(_trending_market(), _spec(symbols=['SA', 'ZZZ']))
    row = {r.symbol: r for r in report.rows}['ZZZ']
    assert (row.target, row.delta, row.tradable, row.contract) == (0, 0, False, '')


def test_an_empty_market_is_an_error_not_an_empty_signal():
    with pytest.raises(ValueError, match='last bar'):
        compute_signal(build_market({}, 0), _spec(symbols=[]))


# ----------------------------------------------------------------------
# Stops
# ----------------------------------------------------------------------

def test_a_resting_stop_is_reported():
    """A stop armed at the last OPEN and not hit is an order that should be
    resting in the real account tomorrow, so it belongs in the signal."""

    class _StoppedLong(Strategy):
        params = {'lots': 1}

        def setup(self, ctx):
            for sym in ctx.symbols:
                ctx.add_indicator('c', sym, ctx.close(sym))

        def on_bar(self, ctx):
            for sym in ctx.symbols:
                if ctx.can_trade(sym):
                    ctx.set_target(sym, 1)
                    ctx.set_stop(sym, distance=5.0)

    report = compute_signal(_trending_market(), _spec(cls=_StoppedLong))
    row = report.rows[0]
    assert row.stop is not None and row.stop_contract == 'SA509'
    assert 'stop' in render(report).lower()


def test_a_resting_take_profit_is_reported():
    """Same for the bracket's other leg: both levels have to reach the desk."""

    class _BracketedLong(Strategy):
        params = {'lots': 1}

        def setup(self, ctx):
            for sym in ctx.symbols:
                ctx.add_indicator('c', sym, ctx.close(sym))

        def on_bar(self, ctx):
            for sym in ctx.symbols:
                if ctx.can_trade(sym):
                    ctx.set_target(sym, 1)
                    ctx.set_stop(sym, distance=5.0)
                    ctx.set_take_profit(sym, distance=20.0)

    report = compute_signal(_trending_market(), _spec(cls=_BracketedLong))
    row = report.rows[0]
    assert row.take_profit is not None and row.take_profit_contract == 'SA509'
    assert row.take_profit > row.stop        # long: the target sits above the stop
    assert 'take profit' in render(report).lower()
    assert report.to_dict()['signals'][0]['take_profit'] == row.take_profit


# ----------------------------------------------------------------------
# The meta arm: same rows, plus an explanation
# ----------------------------------------------------------------------

def test_a_veto_is_explained_in_the_row():
    market = _trending_market()
    report = compute_signal(market, _spec(model=_FakeModel(0.0), model_path='fake.joblib'))
    assert report.is_meta
    vetoed = [r for r in report.rows if r.verdict == 'vetoed']
    assert vetoed, 'a reject-everything model must veto DoubleMa entries'
    for row in vetoed:
        assert row.proba == 0.0 and row.threshold == 0.5
        assert row.vetoed_target is not None
        # The veto is the row's own target, not a note bolted on beside it.
        assert row.target != row.vetoed_target


def test_the_gate_changes_the_signal_and_nothing_else_about_the_shape():
    market = _trending_market()
    plain = compute_signal(market, _spec())
    gated = compute_signal(market, _spec(model=_FakeModel(0.0), model_path='fake.joblib'))
    assert [r.symbol for r in plain.rows] == [r.symbol for r in gated.rows]
    assert any(p.target != g.target for p, g in zip(plain.rows, gated.rows))


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def test_meta_columns_appear_only_when_a_model_is_present():
    market = _trending_market()
    plain = render(compute_signal(market, _spec()))
    gated = render(compute_signal(market, _spec(model=_FakeModel(1.0), model_path='m.joblib')))
    for column in ('proba', 'thresh', 'side'):
        assert column not in plain
        assert column in gated
    for column in ('sym', 'contract', 'sim pos', 'target', 'action', 'stop'):
        assert column in plain and column in gated
    assert 'm.joblib' in gated


def test_the_report_says_which_number_is_actionable():
    text = render(compute_signal(_trending_market(), _spec()))
    assert 'target' in text and 'WILL drift' in text


def test_equity_sized_strategies_are_warned_about_simulated_equity():
    """lots=0 means the lot counts came off the *simulated* equity, which is
    not the account's -- the one caveat that silently changes order sizes."""
    market = _trending_market()
    assert 'SIMULATED equity' in render(compute_signal(market, _spec(params={'lots': 0})))
    assert 'SIMULATED equity' not in render(compute_signal(market, _spec()))


def test_json_is_named_per_strategy_and_date(tmp_path):
    report = compute_signal(_trending_market(), _spec())
    path = save(report, str(tmp_path))
    assert path.endswith(f'signal_DoubleMaStrategy_{report.as_of}.json')
    payload = json.loads(open(path, encoding='utf-8').read())
    assert payload['execute_at'] == 'next session open'
    assert 'model' not in payload
    assert [s['symbol'] for s in payload['signals']] == ['SA', 'CF']
    assert all('proba' not in s for s in payload['signals'])


def test_json_records_the_model_when_there_is_one(tmp_path):
    report = compute_signal(
        _trending_market(), _spec(model=_FakeModel(1.0), model_path='m.joblib'),
    )
    payload = json.loads(open(save(report, str(tmp_path)), encoding='utf-8').read())
    assert payload['model']['path'] == 'm.joblib'
    assert all('proba' in s for s in payload['signals'])


# ----------------------------------------------------------------------
# CLI wiring
# ----------------------------------------------------------------------

def test_model_runs_refuse_primary_flags():
    """--param alongside --model would score today's bar with a model trained
    on a different primary. That must be an error, not a silent mismatch."""
    import live_runner

    args = live_runner._parse_args(['--model', 'm.joblib', '--param', 'fast_period=3'])
    with pytest.raises(ValueError, match='--model'):
        live_runner.build_spec(args)


def test_strategy_runs_resolve_params_and_universe():
    import live_runner

    args = live_runner._parse_args(
        ['--strategy', 'double_ma', '--symbols', 'SA', 'CF', '--param', 'fast_period=3'],
    )
    spec = live_runner.build_spec(args)
    assert spec.strategy_cls is DoubleMaStrategy
    assert spec.params == {'fast_period': 3}
    assert spec.symbols == ['SA', 'CF']
    assert spec.effective_params['slow_period'] == 20   # class default merged under
    assert not spec.is_meta


def test_strategy_and_model_are_mutually_exclusive():
    import live_runner

    with pytest.raises(SystemExit):
        live_runner._parse_args(['--strategy', 'double_ma', '--model', 'm.joblib'])
    with pytest.raises(SystemExit):
        live_runner._parse_args([])
