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
    eng._arm_brackets(0, datetime.date(2024, 1, 1))
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
    eng._arm_brackets(0, datetime.date(2024, 1, 1))

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


def test_a_bracket_with_neither_price_nor_distance_is_refused():
    """Both arguments default to None because either alone is a complete
    instruction. Neither is not, and it used to arm nothing at all -- a
    mistyped keyword left the position unprotected and said nothing."""
    contracts = {'SA509': {0: (100, 101, 99, 100, 100, 0, 10)}}
    panel = build_panel('SA', 1, weighted={'session': [1.0]}, contracts=contracts,
                        contract_by_bar=['SA509'])
    eng = _engine(panel, 1)
    ctx = BarContext(eng, 0, datetime.date(2024, 1, 1))

    with pytest.raises(ValueError, match='price= or distance='):
        ctx.set_stop('SA')
    with pytest.raises(ValueError, match='price= or distance='):
        ctx.set_take_profit('SA')
    assert 'SA' not in eng.stop_spec and 'SA' not in eng.tp_spec

    # Either one on its own still arms, unchanged.
    ctx.set_stop('SA', distance=5.0)
    ctx.set_take_profit('SA', price=110.0)
    assert eng.stop_spec['SA'] == {'distance': 5.0}
    assert eng.tp_spec['SA'] == {'price': 110.0}


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
    eng._arm_brackets(0, datetime.date(2024, 1, 1))

    eng._open_phase(1, datetime.date(2024, 1, 2))
    eng._open_phase(2, datetime.date(2024, 1, 3))   # rolls SA505 -> SA509 here
    assert eng.live_stop['SA']['contract'] == 'SA509'

    eng._open_phase(3, datetime.date(2024, 1, 4))
    eng._intrabar_phase(3, datetime.date(2024, 1, 4))
    assert eng.broker.net_position('SA') == 0        # stop actually fired on the new contract
    assert list(eng.broker.positions) == []          # no leftover phantom position on SA505
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(90.0)


def _bracketed(contracts, n, *, lots=1, stop=None, target=None):
    """Open ``lots`` at bar0's open, arm the requested bracket legs, return eng."""
    panel = build_panel('SA', n, weighted={'session': [1.0] * n}, contracts=contracts,
                        contract_by_bar=['SA509'] * n)
    eng = _engine(panel, n)
    eng.pending['SA'] = lots
    eng._open_phase(0, datetime.date(2024, 1, 1))
    if stop is not None:
        eng.stop_spec['SA'] = {'price': stop}
    if target is not None:
        eng.tp_spec['SA'] = {'price': target}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))
    return eng


def _step(eng, i):
    date = datetime.date(2024, 1, 1 + i)
    eng._open_phase(i, date)
    eng._intrabar_phase(i, date)


def test_take_profit_arms_above_entry_for_a_long_and_below_for_a_short():
    contracts = {'SA509': {0: (100, 105, 99, 100, 100, 0, 10)}}
    long_eng = _bracketed(contracts, 1, lots=1)
    long_eng.tp_spec['SA'] = {'distance': 8.0}
    long_eng._arm_brackets(0, datetime.date(2024, 1, 1))
    assert long_eng.live_tp['SA']['price'] == pytest.approx(108.0)

    short_eng = _bracketed(contracts, 1, lots=-1)
    short_eng.tp_spec['SA'] = {'distance': 8.0}
    short_eng._arm_brackets(0, datetime.date(2024, 1, 1))
    assert short_eng.live_tp['SA']['price'] == pytest.approx(92.0)


def test_take_profit_fills_at_the_target_when_touched_not_gapped():
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        1: (102, 115, 101, 112, 112, 0, 10),   # open (102) short of target (110); high (115) reaches it
    }}
    eng = _bracketed(contracts, 2, target=110.0)
    _step(eng, 1)

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(110.0)
    assert eng.ledger.trades[0]['exit_reason'] == 'take_profit'


def test_take_profit_fills_at_open_when_gapped_through():
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        1: (120, 125, 118, 122, 122, 0, 10),   # opens already above the target (110)
    }}
    eng = _bracketed(contracts, 2, target=110.0)
    _step(eng, 1)

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(120.0)   # today's open, not the target


def test_short_take_profit_fills_at_the_target_when_touched():
    contracts = {'SA509': {
        0: (100, 101, 95, 100, 100, 0, 10),
        1: (98, 99, 85, 88, 88, 0, 10),        # open (98) short of target (90); low (85) reaches it
    }}
    eng = _bracketed(contracts, 2, lots=-1, target=90.0)
    _step(eng, 1)

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(90.0)
    assert eng.ledger.trades[0]['exit_reason'] == 'take_profit'


def test_stop_wins_when_one_bar_touches_both_legs():
    # The open sits between the two levels, so the daily bar cannot say which
    # came first. The pessimistic read is the stop.
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        1: (100, 115, 88, 112, 112, 0, 10),    # reaches both stop (90) and target (110)
    }}
    eng = _bracketed(contracts, 2, stop=90.0, target=110.0)
    _step(eng, 1)

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(90.0)
    assert eng.ledger.trades[0]['exit_reason'] == 'stop'


def test_short_stop_wins_when_one_bar_touches_both_legs():
    contracts = {'SA509': {
        0: (100, 101, 95, 100, 100, 0, 10),
        1: (100, 112, 85, 88, 88, 0, 10),      # reaches both stop (110) and target (90)
    }}
    eng = _bracketed(contracts, 2, lots=-1, stop=110.0, target=90.0)
    _step(eng, 1)

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(110.0)
    assert eng.ledger.trades[0]['exit_reason'] == 'stop'


def test_take_profit_wins_when_the_open_gaps_past_it():
    # Stop-first only applies to an intrabar touch. A long that gaps open above
    # its target never traded back down to the stop first, so ranking the stop
    # ahead here would book a loss on the day the position won.
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        1: (120, 125, 88, 95, 95, 0, 10),      # opens above target (110), then sinks past stop (90)
    }}
    eng = _bracketed(contracts, 2, stop=90.0, target=110.0)
    _step(eng, 1)

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(120.0)
    assert eng.ledger.trades[0]['exit_reason'] == 'take_profit'


def test_a_filled_stop_cancels_the_resting_take_profit():
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        1: (98, 99, 88, 92, 92, 0, 10),        # trips the stop (90), leaves the target (110) untouched
    }}
    eng = _bracketed(contracts, 2, stop=90.0, target=110.0)
    assert eng.live_tp['SA']['price'] == pytest.approx(110.0)
    _step(eng, 1)

    assert eng.broker.net_position('SA') == 0
    # OCO: one leg filling flattens the position, so neither leg survives.
    assert 'SA' not in eng.live_tp and 'SA' not in eng.tp_spec
    assert 'SA' not in eng.live_stop and 'SA' not in eng.stop_spec


def test_a_dark_bar_cannot_fill_a_take_profit_and_the_rule_carries_over():
    contracts = {'SA509': {
        0: (100, 105, 99, 100, 100, 0, 10),
        2: (100, 115, 99, 112, 112, 0, 10),    # first live session after the gap reaches the target
    }}
    panel = build_panel('SA', 3, weighted={'session': [1.0, 0.0, 1.0]}, contracts=contracts,
                        contract_by_bar=['SA509', '', 'SA509'])
    eng = _engine(panel, 3)
    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.tp_spec['SA'] = {'price': 110.0}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))

    _step(eng, 1)                              # dark: no row to check the target against
    assert eng.broker.net_position('SA') == 1
    assert eng.tp_spec['SA'] == {'price': 110.0}

    _step(eng, 2)
    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(110.0)


def test_roll_migrates_the_live_take_profit_to_the_new_contract():
    n = 4
    old = {i: (100.0, 100.0, 100.0, 100.0, 100.0, 0, 10) for i in range(n)}
    new = {i: (100.0, 100.0, 100.0, 100.0, 100.0, 0, 10) for i in range(2, n)}
    new[3] = (100.0, 115.0, 100.0, 112.0, 112.0, 0, 10)   # reaches the target after the roll
    panel = build_panel(
        'SA', n, weighted={'session': [1.0] * n},
        contracts={'SA505': old, 'SA509': new},
        contract_by_bar=['SA505', 'SA505', 'SA509', 'SA509'],
    )
    eng = _engine(panel, n)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.tp_spec['SA'] = {'price': 110.0}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))

    eng._open_phase(1, datetime.date(2024, 1, 2))
    eng._open_phase(2, datetime.date(2024, 1, 3))   # rolls SA505 -> SA509 here
    assert eng.live_tp['SA']['contract'] == 'SA509'

    eng._open_phase(3, datetime.date(2024, 1, 4))
    eng._intrabar_phase(3, datetime.date(2024, 1, 4))
    assert eng.broker.net_position('SA') == 0
    assert list(eng.broker.positions) == []
    assert eng.ledger.trades[0]['close_price'] == pytest.approx(110.0)


def test_take_profit_rearms_after_a_close_and_reopen():
    # bar0: buy + target@110. bar2: close (flat by bar3 open). bar3: reopen with
    # a fresh target@105. bar5 reaches 106: only trips if the *new* target is
    # armed -- the stale 110 from the first leg would not.
    n = 6
    contracts = {'SA509': {
        0: (100, 101, 99, 100, 100, 0, 10),
        1: (100, 101, 99, 100, 100, 0, 10),
        2: (100, 101, 99, 100, 100, 0, 10),
        3: (100, 101, 99, 100, 100, 0, 10),
        4: (100, 101, 99, 100, 100, 0, 10),
        5: (100, 106, 99, 104, 104, 0, 10),
    }}
    panel = build_panel('SA', n, weighted={'session': [1.0] * n}, contracts=contracts,
                        contract_by_bar=['SA509'] * n)

    class _TpScripted(Strategy):
        def setup(self, ctx):
            pass

        def on_bar(self, ctx):
            if ctx.i == 0:
                ctx.set_target('SA', 1)
                ctx.set_take_profit('SA', price=110.0)
            elif ctx.i == 2:
                ctx.close('SA')
                ctx.cancel_take_profit('SA')
            elif ctx.i == 3:
                ctx.set_target('SA', 1)
                ctx.set_take_profit('SA', price=105.0)

    md = build_market({'SA': panel}, n)
    eng = Engine(md, _TpScripted(), initial_cash=100_000.0, slippage=0.0)
    eng.run_backtest(SetupContext, BarContext)

    assert eng.broker.net_position('SA') == 0
    assert eng.ledger.trades[-1]['close_price'] == pytest.approx(105.0)
    assert eng.ledger.trades[-1]['exit_reason'] == 'take_profit'


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


def test_set_target_nets_out_an_order_still_deferred_from_a_dark_bar():
    # bar1 asks for +2, bar2 is dark so that order is held over, and bar2's
    # own signal cancels it back to flat. The deferred lots replay at bar3's
    # open ahead of anything queued, so a target measured off `net_position`
    # alone queues nothing and the run ends long 2 -- the opposite of the
    # strategy's last instruction. Netting the in-flight amount out cancels
    # the two against each other and nothing fills at all.
    n = 5
    price = [100.0, 101.0, 102.0, 103.0, 104.0]
    contracts = {'SA509': {
        i: (price[i], price[i] + 1, price[i] - 1, price[i], price[i], 5000, 900)
        for i in range(n) if i != 2
    }}
    panel = build_panel(
        'SA', n,
        weighted={'session': [1.0, 1.0, 0.0, 1.0, 1.0], 'close': price},
        contracts=contracts, contract_by_bar=['SA509', 'SA509', '', 'SA509', 'SA509'],
    )
    md = build_market({'SA': panel}, n)
    eng = Engine(md, _ScriptedStrategy({1: 2, 2: 0}), initial_cash=100_000.0)
    eng.run_backtest(SetupContext, BarContext)

    assert eng.broker.net_position('SA') == 0
    assert eng.signal_log == []
    assert eng.deferred.get('SA', 0) == 0


def test_set_target_still_reaches_a_new_target_over_a_deferred_order():
    # Same dark bar, but bar2 raises the target to 3 instead of cancelling.
    # Netting must queue only the missing lot, not a fresh 3 on top of the
    # 2 already in flight.
    n = 5
    price = [100.0, 101.0, 102.0, 103.0, 104.0]
    contracts = {'SA509': {
        i: (price[i], price[i] + 1, price[i] - 1, price[i], price[i], 5000, 900)
        for i in range(n) if i != 2
    }}
    panel = build_panel(
        'SA', n,
        weighted={'session': [1.0, 1.0, 0.0, 1.0, 1.0], 'close': price},
        contracts=contracts, contract_by_bar=['SA509', 'SA509', '', 'SA509', 'SA509'],
    )
    md = build_market({'SA': panel}, n)
    eng = Engine(md, _ScriptedStrategy({1: 2, 2: 3}), initial_cash=100_000.0)
    eng.run_backtest(SetupContext, BarContext)

    assert eng.broker.net_position('SA') == 3
    assert len(eng.signal_log) == 1          # one fill of +3, not +2 then +3
    assert eng.signal_log[0]['size'] == 3


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


# --------------------------------------------------------------------------
# Slippage, charged in ticks
# --------------------------------------------------------------------------

def _slippage_engine(symbol, slippage, contract, n_bars=2):
    bars = {i: (100, 101, 99, 100, 100, 0, 10) for i in range(n_bars)}
    panel = build_panel(symbol, n_bars, weighted={'session': [1.0] * n_bars},
                        contracts={contract: bars},
                        contract_by_bar=[contract] * n_bars)
    md = build_market({symbol: panel}, n_bars)
    return Engine(md, _NullStrategy(), initial_cash=1_000_000.0, slippage=slippage)


def test_slippage_scales_with_the_product_tick_not_the_price():
    """The same setting costs one tick on each product, so it is portable.

    SA ticks at 1.0 and CF at 5.0; both open at 100 here, so any difference in
    the fill price is the tick and nothing else.
    """
    sa = _slippage_engine('SA', 1.0, 'SA509')
    sa.pending['SA'] = 1
    sa._open_phase(0, datetime.date(2024, 1, 1))
    assert sa.signal_log[-1]['price'] == pytest.approx(101.0)   # 100 + 1 x 1.0

    cf = _slippage_engine('CF', 1.0, 'CF509')
    cf.pending['CF'] = 1
    cf._open_phase(0, datetime.date(2024, 1, 1))
    assert cf.signal_log[-1]['price'] == pytest.approx(105.0)   # 100 + 1 x 5.0


def test_slippage_is_paid_in_the_direction_of_the_trade():
    eng = _slippage_engine('CF', 2.0, 'CF509')
    eng.pending['CF'] = -1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    assert eng.signal_log[-1]['price'] == pytest.approx(90.0)   # 100 - 2 x 5.0


def test_zero_slippage_fills_exactly_at_the_open():
    eng = _slippage_engine('CF', 0.0, 'CF509')
    eng.pending['CF'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    assert eng.signal_log[-1]['price'] == pytest.approx(100.0)


def test_a_roll_pays_slippage_on_both_legs():
    """Entry, roll and exit each cross the spread, so the round trip pays four
    ticks of slippage in total -- two of them inside the roll."""
    old = {i: (100, 101, 99, 100, 100, 0, 10) for i in range(3)}
    new = {i: (200, 201, 199, 200, 200, 0, 10) for i in range(3)}
    panel = build_panel('SA', 3, weighted={'session': [1.0, 1.0, 1.0]},
                        contracts={'SA505': old, 'SA509': new},
                        contract_by_bar=['SA505', 'SA509', 'SA509'])
    md = build_market({'SA': panel}, 3)
    eng = Engine(md, _NullStrategy(), initial_cash=1_000_000.0, slippage=3.0)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))       # buys SA505 at 100 + 3
    eng._open_phase(1, datetime.date(2024, 1, 2))       # rolls: sells 100 - 3, buys 200 + 3
    eng.pending['SA'] = -1
    eng._open_phase(2, datetime.date(2024, 1, 3))       # sells SA509 at 200 - 3

    trade = eng.ledger.trades[0]
    assert trade['open_price'] == pytest.approx(103.0)
    assert trade['close_price'] == pytest.approx(197.0)
    assert trade['n_rolls'] == 1
    # -6 points on each leg, at SA's multiplier of 20
    assert trade['gross_pnl'] == pytest.approx(-240.0)


# --------------------------------------------------------------------------
# Margin rejection and blow-up
# --------------------------------------------------------------------------

def _cf_engine(cash, n_bars=3, price=15_000.0, closes=None):
    """One CF product printing a flat ``price``, so only sizing is in play."""
    rows = {i: (price, price, price, price, price, 0, 10) for i in range(n_bars)}
    if closes:
        for i, p in closes.items():
            rows[i] = (p, p, p, p, p, 0, 10)
    panel = build_panel('CF', n_bars, weighted={'session': [1.0] * n_bars},
                        contracts={'CF509': rows},
                        contract_by_bar=['CF509'] * n_bars)
    md = build_market({'CF': panel}, n_bars)
    return Engine(md, _NullStrategy(), initial_cash=cash, slippage=0.0)


def test_an_unaffordable_order_is_rejected_not_filled():
    """CF needs 7,500 margin per lot at 15,000; 10,000 cash funds one, not two."""
    eng = _cf_engine(10_000.0)
    eng.pending['CF'] = 2
    eng._open_phase(0, datetime.date(2024, 1, 1))

    assert eng.broker.net_position('CF') == 0
    assert len(eng.rejections) == 1
    assert eng.rejections[0]['symbol'] == 'CF'
    assert eng.rejections[0]['size'] == 2
    assert eng.broker.cash == 10_000.0          # no commission on a refused order


def test_a_rejected_order_is_not_replayed_on_the_next_bar():
    """Rejected, not deferred -- the distinction a dark bar makes the other way."""
    eng = _cf_engine(10_000.0)
    eng.pending['CF'] = 2
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng._open_phase(1, datetime.date(2024, 1, 2))

    assert eng.broker.net_position('CF') == 0
    assert eng.deferred.get('CF', 0) == 0
    assert len(eng.rejections) == 1              # refused once, then gone


def test_an_affordable_order_still_fills_normally():
    eng = _cf_engine(10_000.0)
    eng.pending['CF'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))

    assert eng.broker.net_position('CF') == 1
    assert eng.rejections == []


def test_a_stretched_account_can_still_close_what_it_cannot_add_to():
    """8,000 funds exactly one 7,500-margin lot: adding is refused, exiting is not."""
    eng = _cf_engine(8_000.0)
    eng.pending['CF'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    assert eng.broker.net_position('CF') == 1

    eng.pending['CF'] = 1
    eng._open_phase(1, datetime.date(2024, 1, 2))
    assert eng.broker.net_position('CF') == 1            # the 2nd lot is refused
    assert len(eng.rejections) == 1

    eng.pending['CF'] = -1
    eng._open_phase(2, datetime.date(2024, 1, 3))
    assert eng.broker.net_position('CF') == 0            # the exit is not
    assert len(eng.rejections) == 1


class _AlwaysLong(Strategy):
    """Wants one lot of everything, every bar, regardless of capital."""

    def setup(self, ctx):
        pass

    def on_bar(self, ctx):
        for sym in ctx.symbols:
            if ctx.can_trade(sym):
                ctx.set_target(sym, 1)


def test_a_collapse_blows_the_account_up_and_stops_the_run():
    """A gap far enough down to wipe out equity outright.

    The forced liquidation fires first and still leaves nothing, so the run
    stops on that bar rather than trading the two after it on capital that no
    longer exists.
    """
    # bar0 signals, bar1 fills at 15,000, bar2 opens and settles at 1,000
    prices = [15_000.0, 15_000.0, 1_000.0, 1_000.0]
    rows = {i: (p, p, p, p, p, 0, 10) for i, p in enumerate(prices)}
    panel = build_panel('CF', 4, weighted={'session': [1.0] * 4},
                        contracts={'CF509': rows}, contract_by_bar=['CF509'] * 4)
    md = build_market({'CF': panel}, 4)
    eng = Engine(md, _AlwaysLong(), initial_cash=10_000.0, slippage=0.0)
    out = eng.run_backtest(SetupContext, BarContext)

    assert out['blown_up'] is True
    assert out['blown_up_date'] == datetime.date(2024, 1, 3)    # bar 2
    assert len(out['equity_records']) == 3                       # bar 3 never ran
    assert out['equity_records'][-1]['equity'] <= 0
    assert eng.broker.positions == {}                            # liquidated on the way out


def test_a_solvent_run_is_not_marked_blown_up():
    prices = [15_000.0] * 4
    rows = {i: (p, p, p, p, p, 0, 10) for i, p in enumerate(prices)}
    panel = build_panel('CF', 4, weighted={'session': [1.0] * 4},
                        contracts={'CF509': rows}, contract_by_bar=['CF509'] * 4)
    md = build_market({'CF': panel}, 4)
    eng = Engine(md, _AlwaysLong(), initial_cash=1_000_000.0, slippage=0.0)
    out = eng.run_backtest(SetupContext, BarContext)

    assert out['blown_up'] is False
    assert out['blown_up_date'] is None
    assert len(out['equity_records']) == 4


# ----------------------------------------------------------------------
# Rolls: one leg per product, and brackets that survive the basis
# ----------------------------------------------------------------------
#
# All four of these describe the same failure from different angles: a roll
# that could not complete on its own bar, and everything that used to go
# wrong while it waited.


def _level(live, sym):
    """The (price, contract) a resting bracket leg is on -- the part a test
    cares about, without the `anchor` bookkeeping `_arm_brackets` uses to
    decide when to re-resolve it."""
    order = live[sym]
    return order['price'], order['contract']


def _rolling_panel(old_bars, new_bars, n, *, old_price=100.0, new_price=100.0, roll_at=2):
    """SA on SA505 until ``roll_at``, on SA509 after. ``*_bars`` list which
    bars each contract actually printed on, so a test can make one go dark."""
    def rows(bars, p):
        return {i: (p, p + 1.0, p - 1.0, p, p, 0, 10) for i in bars}
    return build_panel(
        'SA', n, weighted={'session': [1.0] * n},
        contracts={'SA505': rows(old_bars, old_price), 'SA509': rows(new_bars, new_price)},
        contract_by_bar=['SA505'] * roll_at + ['SA509'] * (n - roll_at),
    )


def test_a_pending_order_never_opens_a_second_leg_while_a_roll_waits():
    """SA505 is dark on the roll bar, so its roll has to wait -- and the order
    queued for that same bar must wait with it.

    Filling it on the calendar contract instead would leave the product
    holding SA505 *and* SA509 at once, which is not a position any strategy
    asked for: brackets, the ledger and the margin model are all keyed by
    product and each would then describe only half the book.
    """
    n = 6
    panel = _rolling_panel([0, 1, 3, 4, 5], list(range(n)), n)   # SA505 dark on bar 2
    eng = _engine(panel, n)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng._open_phase(1, datetime.date(2024, 1, 2))
    eng.pending['SA'] = 1                                        # queued for the roll bar
    eng._open_phase(2, datetime.date(2024, 1, 3))                # roll waits; so does the order

    assert list(eng.broker.positions) == [('SA', 'SA505')]       # still one leg
    assert eng.deferred['SA'] == 1                               # order held, not misrouted

    eng._open_phase(3, datetime.date(2024, 1, 4))                # SA505 prints: roll, then fill
    assert list(eng.broker.positions) == [('SA', 'SA509')]
    assert eng.broker.net_position('SA') == 2
    assert not eng.deferred.get('SA')


def test_a_delayed_roll_does_not_compound_the_position_it_is_waiting_on():
    """The regression proper: rolling ``net_position`` on a *single* leg.

    Closing the product's net across contracts on one contract overshoots it
    the moment there are two, flipping that leg short and re-opening the whole
    net on the target -- every bar, without bound. Twelve bars of it turned a
    2-lot position into 36 gross lots and 18x the margin.
    """
    n = 12
    panel = _rolling_panel([i for i in range(n) if i != 2], list(range(n)), n)
    eng = _engine(panel, n)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.pending['SA'] = 1
    for i in range(1, n):
        eng._open_phase(i, datetime.date(2024, 1, 1) + datetime.timedelta(days=i))

    gross = sum(abs(p.size) for p in eng.broker.positions.values())
    assert eng.broker.net_position('SA') == 2
    assert gross == 2                                   # not 36
    _, margin_used, _ = eng.broker.valuation()
    assert margin_used == pytest.approx(2 * 100.0 * 20 * 0.11)


def test_a_contract_that_never_prints_again_is_rolled_at_its_carried_mark():
    """A leg whose contract has expired cannot be waited on -- there is no
    future print to roll it at. Left alone it stayed pinned to that contract
    for the rest of the run, frozen at a stale mark, and was finally closed at
    that stale price. It is moved on instead, and said out loud: the exit is
    at a price nobody traded."""
    n = 5
    panel = _rolling_panel([0, 1], list(range(n)), n, new_price=120.0)   # SA505 dies at bar 1
    eng = _engine(panel, n)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng._open_phase(1, datetime.date(2024, 1, 2))
    eng._open_phase(2, datetime.date(2024, 1, 3))       # SA505 is finished: forced roll

    assert list(eng.broker.positions) == [('SA', 'SA509')]
    assert eng.broker.net_position('SA') == 1
    assert len(eng.stranded_rolls) == 1
    stranded = eng.stranded_rolls[0]
    assert (stranded['from_contract'], stranded['to_contract']) == ('SA505', 'SA509')
    assert stranded['mark'] == pytest.approx(100.0)     # SA505's last real print


def test_a_roll_reanchors_a_distance_stop_against_the_new_contracts_entry():
    # SA505 at 100, SA509 at 90: a 10-point basis, which is what the stop must
    # not inherit. A distance is re-resolved against the new fill.
    n = 4
    panel = _rolling_panel(list(range(n)), list(range(n)), n, old_price=100.0, new_price=90.0)
    eng = _engine(panel, n)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'distance': 5.0}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))
    assert _level(eng.live_stop, 'SA') == (95.0, 'SA505')

    _step(eng, 2)                                        # rolls SA505 -> SA509
    assert eng.broker.positions[('SA', 'SA509')].avg_entry == pytest.approx(90.0)
    assert _level(eng.live_stop, 'SA') == (85.0, 'SA509')
    assert eng.broker.net_position('SA') == 1            # nothing tripped on the way


def test_a_backwardated_roll_shifts_an_explicit_stop_instead_of_tripping_it():
    """The stop level used to migrate contracts but keep its price.

    Rolling from SA505 at 100 into SA509 at 90 then left a long's stop at 95 --
    *above* the market it was now being checked against -- and
    ``_bracket_hit``'s gap tier flattened the position at that same open, for
    a loss the strategy's own rule never called for.
    """
    n = 4
    panel = _rolling_panel(list(range(n)), list(range(n)), n, old_price=100.0, new_price=90.0)
    eng = _engine(panel, n)

    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'price': 95.0}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))

    _step(eng, 2)                                        # rolls, and must not stop out
    assert eng.broker.net_position('SA') == 1
    assert [t['exit_reason'] for t in eng.ledger.trades] == []
    # Shifted by the basis the roll realized, so it sits the same 5 points
    # below the market the strategy chose it against.
    assert _level(eng.live_stop, 'SA') == (85.0, 'SA509')
    assert eng.stop_spec['SA'] == {'price': 85.0}        # sticky spec moved too, not just the live order


# ----------------------------------------------------------------------
# Bracket lifecycle: the level tracks the rule and the position
# ----------------------------------------------------------------------
#
# Four symptoms of one cause. `_arm_brackets` used to re-resolve a leg only
# when no order happened to be resting, which made the level depend on
# unrelated history instead of on what it was resolved from.


def _priced(n, price=100.0, sessions=None, rows=None):
    contracts = {'SA509': rows if rows is not None else
                 {i: (price, price + 1, price - 1, price, price, 0, 10) for i in range(n)}}
    return build_panel('SA', n, weighted={'session': sessions or [1.0] * n},
                       contracts=contracts, contract_by_bar=['SA509'] * n)


def test_a_dark_bar_does_not_move_a_stop():
    """A dark session cleared the resting order, and `_arm_brackets` put it
    back at the end of the same phase -- re-resolved against whatever the
    position's cost had become since. So adding to a position moved the stop
    if a non-trading day happened to follow, and left it alone if none did:
    two runs differing only in where the holidays fell disagreed.
    """
    def run(sessions, rows):
        panel = _priced(6, sessions=sessions, rows=rows)
        eng = _engine(panel, 6)
        eng.pending['SA'] = 1                       # 1 lot @100
        eng._open_phase(0, datetime.date(2024, 1, 1))
        eng.stop_spec['SA'] = {'distance': 5.0}
        eng._arm_brackets(0, datetime.date(2024, 1, 1))
        eng._open_phase(1, datetime.date(2024, 1, 2))
        eng.pending['SA'] = 1                       # add a 2nd lot @120 -> avg_entry 110
        for i in range(2, 6):
            eng._open_phase(i, datetime.date(2024, 1, 1) + datetime.timedelta(days=i))
        return eng.live_stop['SA']['price']

    bar = lambda p: (p, p + 1, p - 1, p, p, 0, 10)
    dense = run([1.0] * 6, {i: bar(100 if i < 2 else 120) for i in range(6)})
    with_gap = run([1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                   {0: bar(100), 1: bar(100), 2: bar(120), 4: bar(120), 5: bar(120)})
    assert dense == with_gap == pytest.approx(105.0)   # 110 - 5, whichever calendar


def test_set_stop_moves_a_stop_that_is_already_resting():
    """`set_stop` wrote the sticky spec but `_arm_brackets` skipped re-arming
    while an order rested, so the new level did nothing until the position
    closed and reopened -- which made a trailing stop impossible to express.
    """
    eng = _engine(_priced(4), 4)
    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'price': 90.0}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))
    assert eng.live_stop['SA']['price'] == pytest.approx(90.0)

    BarContext(eng, 1, datetime.date(2024, 1, 2)).set_stop('SA', price=97.0)
    eng._open_phase(2, datetime.date(2024, 1, 3))
    assert eng.live_stop['SA']['price'] == pytest.approx(97.0)


def test_a_reversal_puts_the_stop_on_the_new_sides_of_the_market():
    """The worst of the four. Flipping long -> short left the long's stop
    resting *below* the market; `_bracket_hit`'s short branch reads a stop at
    or under the open as gapped through, so the fresh short was flattened at
    the very open that opened it.
    """
    eng = _engine(_priced(4), 4)
    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'distance': 5.0}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))
    assert eng.live_stop['SA']['price'] == pytest.approx(95.0)      # long: below entry

    eng.pending['SA'] = -2                                          # set_target(-1): one order
    _step(eng, 1)
    assert eng.broker.net_position('SA') == -1                      # the short survived the bar
    assert eng.live_stop['SA']['price'] == pytest.approx(105.0)     # short: above entry


def test_cancelling_a_spec_takes_its_resting_order_with_it():
    eng = _engine(_priced(4), 4)
    eng.pending['SA'] = 1
    eng._open_phase(0, datetime.date(2024, 1, 1))
    eng.stop_spec['SA'] = {'price': 90.0}
    eng._arm_brackets(0, datetime.date(2024, 1, 1))
    assert 'SA' in eng.live_stop

    eng.stop_spec.pop('SA')            # the spec is the source of truth
    eng._open_phase(1, datetime.date(2024, 1, 2))
    assert 'SA' not in eng.live_stop
