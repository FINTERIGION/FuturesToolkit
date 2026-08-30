import datetime

import pytest

from core.engine import Engine
from strategies.base import BarContext, SetupContext, Strategy
from tests.conftest import build_market, build_panel


class _NullStrategy(Strategy):
    def setup(self, ctx):
        pass

    def on_bar(self, ctx):
        pass


def _engine(panel, n_bars):
    md = build_market({'SA': panel}, n_bars)
    return Engine(md, _NullStrategy(), initial_cash=100_000.0, slippage=0.0)


def test_dark_bar_defers_order_to_next_live_session():
    # bar0 live (open=100), bar1 dark (no print), bar2 live again (open=110)
    session = [1.0, 0.0, 1.0]
    contract_by_bar = ['SA509', '', 'SA509']
    contracts = {'SA509': {
        0: (100, 101, 99, 100, 100, 0, 10),
        2: (110, 111, 109, 110, 110, 0, 10),
    }}
    panel = build_panel('SA', 3, weighted={'session': session}, contracts=contracts, contract_by_bar=contract_by_bar)
    eng = _engine(panel, 3)

    eng.pending['SA'] = 2
    eng._open_phase(0, datetime.date(2024, 1, 1))
    assert eng.broker.net_position('SA') == 2

    eng.pending['SA'] = -1
    eng._open_phase(1, datetime.date(2024, 1, 2))
    assert eng.broker.net_position('SA') == 2       # bar1 is dark: not filled
    assert eng.deferred['SA'] == -1
    assert 'SA' not in eng.pending

    eng._open_phase(2, datetime.date(2024, 1, 3))
    assert eng.broker.net_position('SA') == 1        # replays at bar2's live open
    assert eng.deferred.get('SA', 0) == 0


def test_stop_fills_same_bar_at_stop_price_when_touched_not_gapped():
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        1: (98, 99, 90, 92, 92, 0, 10),     # open (98) hasn't gapped past stop (95); low (90) touches it
    }}
    panel = build_panel('SA', 2, weighted={'session': [1.0, 1.0]}, contracts=contracts,
                         contract_by_bar=['SA509', 'SA509'])
    eng = _engine(panel, 2)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'distance': 5.0}
    eng._arm_stops(0, datetime.date(2024, 1, 1))
    assert eng.live_stop['SA']['price'] == pytest.approx(95.0)

    eng._open_phase(1, datetime.date(2024, 1, 2))
    eng._intrabar_phase(1, datetime.date(2024, 1, 2))

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(95.0)


def test_stop_fills_at_open_when_gapped_through():
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        1: (80, 85, 78, 82, 82, 0, 10),      # opens already below the stop (95)
    }}
    panel = build_panel('SA', 2, weighted={'session': [1.0, 1.0]}, contracts=contracts,
                         contract_by_bar=['SA509', 'SA509'])
    eng = _engine(panel, 2)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'distance': 5.0}
    eng._arm_stops(0, datetime.date(2024, 1, 1))

    eng._open_phase(1, datetime.date(2024, 1, 2))
    eng._intrabar_phase(1, datetime.date(2024, 1, 2))

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(80.0)   # today's open, not the stop price


class _ScriptedStrategy(Strategy):
    def __init__(self, script):
        super().__init__()
        self.script = script   # {bar_index: target_lots}

    def setup(self, ctx):
        pass

    def on_bar(self, ctx):
        target = self.script.get(ctx.i)
        if target is not None:
            ctx.set_target('SA', target)


class _StopScriptedStrategy(Strategy):
    """{bar_index: (target_lots, stop_price_or_None)} driven via set_target/set_stop."""

    def __init__(self, script):
        super().__init__()
        self.script = script

    def setup(self, ctx):
        pass

    def on_bar(self, ctx):
        step = self.script.get(ctx.i)
        if step is None:
            return
        target, stop_price = step
        ctx.set_target('SA', target)
        if stop_price is not None:
            ctx.set_stop('SA', price=stop_price)


def test_stop_is_rearmed_after_a_close_and_reopen():
    # bar0: buy + stop@90. bar2: close (flat by bar3 open). bar3: reopen with
    # a fresh stop@99. bar5 dips to low=95: only trips if the *new* stop (99)
    # is actually armed -- the stale 90 from the first leg would not.
    n = 6
    rows = {i: (100.0, 100.0, 100.0, 100.0, 100.0, 0, 10) for i in range(n)}
    rows[5] = (100.0, 100.0, 95.0, 97.0, 97.0, 0, 10)
    contracts = {'SA509': rows}
    panel = build_panel('SA', n, weighted={'session': [1.0] * n}, contracts=contracts,
                         contract_by_bar=['SA509'] * n)
    md = build_market({'SA': panel}, n)
    strat = _StopScriptedStrategy({0: (1, 90.0), 2: (0, None), 3: (1, 99.0)})
    eng = Engine(md, strat, initial_cash=100_000.0)
    eng.run_backtest(SetupContext, BarContext)

    assert eng.broker.net_position('SA') == 0
    assert len(eng.ledger.trades) == 2
    assert eng.ledger.trades[-1]['close_price'] == pytest.approx(99.0)


def test_roll_migrates_the_live_stop_to_the_new_contract():
    # position survives an April/August/December roll from SA505 to SA509;
    # the resting stop must move to the new contract, not keep checking the
    # old (now-flat) one.
    n = 4
    old = {i: (100.0, 100.0, 100.0, 100.0, 100.0, 0, 10) for i in range(n)}
    new = {i: (100.0, 100.0, 100.0, 100.0, 100.0, 0, 10) for i in range(2, n)}
    new[3] = (100.0, 100.0, 80.0, 85.0, 85.0, 0, 10)   # dips through the stop after the roll
    panel = build_panel(
        'SA', n, weighted={'session': [1.0] * n},
        contracts={'SA505': old, 'SA509': new},
        contract_by_bar=['SA505', 'SA505', 'SA509', 'SA509'],
    )
    eng = _engine(panel, n)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'price': 90.0}
    eng._arm_stops(0, datetime.date(2024, 1, 1))

    eng._open_phase(1, datetime.date(2024, 1, 2))
    eng._open_phase(2, datetime.date(2024, 1, 3))   # rolls SA505 -> SA509 here
    assert eng.live_stop['SA']['contract'] == 'SA509'

    eng._open_phase(3, datetime.date(2024, 1, 4))
    eng._intrabar_phase(3, datetime.date(2024, 1, 4))
    assert eng.broker.net_position('SA') == 0        # stop actually fired on the new contract
    assert list(eng.broker.positions) == []          # no leftover phantom position on SA505
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(90.0)


def test_set_target_reversal_fills_in_a_single_order():
    n = 4
    contracts = {'SA509': {i: (100 + i, 101 + i, 99 + i, 100 + i, 100 + i, 0, 10) for i in range(n)}}
    panel = build_panel(
        'SA', n,
        weighted={'session': [1.0] * n, 'close': [100.0, 101.0, 102.0, 103.0]},
        contracts=contracts, contract_by_bar=['SA509'] * n,
    )
    md = build_market({'SA': panel}, n)
    strat = _ScriptedStrategy({0: 2, 1: -2})   # bar0 signal fills at bar1 open; bar1 signal fills at bar2 open
    eng = Engine(md, strat, initial_cash=100_000.0)
    eng.run_backtest(SetupContext, BarContext)

    # bar1's open (101) fills the +2 entry; bar2's open (102) fills the -4 reversal -- one order each.
    assert len(eng.signal_log) == 2
    assert eng.broker.net_position('SA') == -2

    assert len(eng.ledger.trades) == 2
    closed, still_open = eng.ledger.trades
    assert closed['direction'] == 'long'
    assert closed['close_price'] == pytest.approx(102.0)
    assert still_open['direction'] == 'short'
    assert still_open['open_price'] == pytest.approx(102.0)
    assert still_open['open_at_end'] == 1


def test_signal_defers_when_the_calendar_contract_has_no_print():
    # session says live and contract_by_bar names SA509, but SA509 itself has
    # no row on bar1 -- a torn calendar/contract pair the engine must not
    # crash on. The order waits for the next bar that does print.
    contracts = {'SA509': {
        0: (100, 101, 99, 100, 100, 0, 10),
        2: (110, 111, 109, 110, 110, 0, 10),   # nothing at bar1
    }}
    panel = build_panel('SA', 3, weighted={'session': [1.0, 1.0, 1.0]}, contracts=contracts,
                         contract_by_bar=['SA509', 'SA509', 'SA509'])
    eng = _engine(panel, 3)

    eng.pending['SA'] = 2
    eng._open_phase(1, datetime.date(2024, 1, 2))
    assert eng.broker.net_position('SA') == 0
    assert eng.deferred['SA'] == 2
    assert 'SA' not in eng.pending

    eng._open_phase(2, datetime.date(2024, 1, 3))
    assert eng.broker.net_position('SA') == 2
    assert eng.signal_log[0]['price'] == pytest.approx(110.0)
    assert eng.deferred.get('SA', 0) == 0


def test_signal_defers_when_the_calendar_names_an_unknown_contract():
    # contract_by_bar points at a code with no ContractSeries at all.
    contracts = {'SA509': {0: (100, 101, 99, 100, 100, 0, 10), 1: (105, 106, 104, 105, 105, 0, 10)}}
    panel = build_panel('SA', 2, weighted={'session': [1.0, 1.0]}, contracts=contracts,
                         contract_by_bar=['SA509', 'SA601'])
    eng = _engine(panel, 2)

    eng.pending['SA'] = 1
    eng._open_phase(1, datetime.date(2024, 1, 2))
    assert eng.broker.net_position('SA') == 0
    assert eng.deferred['SA'] == 1


def test_roll_is_delayed_when_the_target_contract_has_no_print():
    # bar1 wants SA505 -> SA509, but SA509 has no row that day; the roll waits
    # for bar2 rather than pricing off a missing row.
    old = {0: (100, 101, 99, 100, 100, 0, 10), 1: (102, 103, 101, 102, 102, 0, 10),
           2: (104, 105, 103, 104, 104, 0, 10)}
    new = {0: (200, 201, 199, 200, 200, 0, 10), 2: (204, 205, 203, 204, 204, 0, 10)}
    panel = build_panel('SA', 3, weighted={'session': [1.0, 1.0, 1.0]},
                         contracts={'SA505': old, 'SA509': new},
                         contract_by_bar=['SA505', 'SA509', 'SA509'])
    eng = _engine(panel, 3)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    assert eng._current_contract('SA') == 'SA505'

    eng._open_phase(1, datetime.date(2024, 1, 2))
    assert eng._current_contract('SA') == 'SA505'   # still on the old leg
    assert eng.broker.net_position('SA') == 1

    eng._open_phase(2, datetime.date(2024, 1, 3))
    assert eng._current_contract('SA') == 'SA509'
    assert eng.broker.net_position('SA') == 1


def test_roll_is_delayed_when_the_target_contract_is_unknown():
    old = {i: (100, 101, 99, 100, 100, 0, 10) for i in range(2)}
    panel = build_panel('SA', 2, weighted={'session': [1.0, 1.0]}, contracts={'SA505': old},
                         contract_by_bar=['SA505', 'SA601'])
    eng = _engine(panel, 2)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng._open_phase(1, datetime.date(2024, 1, 2))

    assert eng._current_contract('SA') == 'SA505'
    assert eng.broker.net_position('SA') == 1
