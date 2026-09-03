import datetime

import pytest

from core.broker import Broker
from core.ledger import Ledger
from core.types import Reason

D = datetime.date(2024, 1, 1)


def _do(broker, ledger, symbol, contract, size, price, i, reason=Reason.SIGNAL):
    f = broker.fill(symbol, contract, size, price, i, D, reason=reason)
    if f:
        ledger.process_fill(f)
    return f


def test_roll_folds_into_one_logical_trade_both_legs_commissioned():
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'SA', 'SA509', 2, 100.0, 0)                              # open long 2
    _do(b, lg, 'SA', 'SA509', -2, 105.0, 1, reason=Reason.ROLL)          # roll: close old
    _do(b, lg, 'SA', 'SA601', 2, 105.0, 1, reason=Reason.ROLL)           # roll: open new
    _do(b, lg, 'SA', 'SA601', -2, 108.0, 2)                              # close

    assert len(lg.trades) == 1
    trade = lg.trades[0]
    assert trade['n_rolls'] == 1
    assert trade['contracts'] == 'SA509|SA601'
    # gross_pnl = roll-leg realize (105-100)*2*20=200 + final close (108-105)*2*20=120
    assert trade['gross_pnl'] == pytest.approx(320.0)
    # commission on all four legs: 2*100*20e-4 + 2*105*20e-4 + 2*105*20e-4 + 2*108*20e-4
    expected_comm = 2 * 100 * 20 * 1e-4 + 2 * 105 * 20 * 1e-4 + 2 * 105 * 20 * 1e-4 + 2 * 108 * 20 * 1e-4
    assert trade['commission'] == pytest.approx(expected_comm)


def test_exit_reason_records_the_fill_that_flattened_the_trade():
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'SA', 'SA509', 2, 100.0, 0)
    _do(b, lg, 'SA', 'SA509', -2, 112.0, 1, reason=Reason.TAKE_PROFIT)

    assert lg.trades[0]['exit_reason'] == 'take_profit'


def test_exit_reason_ignores_a_scale_out_and_a_roll():
    # Only the fill that takes the position to flat names the exit: a partial
    # reduce partway through, and a roll's closing leg, must not.
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'SA', 'SA509', 4, 100.0, 0)
    _do(b, lg, 'SA', 'SA509', -2, 105.0, 1)                              # scale out half
    _do(b, lg, 'SA', 'SA509', -2, 106.0, 2, reason=Reason.ROLL)
    _do(b, lg, 'SA', 'SA601', 2, 106.0, 2, reason=Reason.ROLL)
    _do(b, lg, 'SA', 'SA601', -2, 90.0, 3, reason=Reason.STOP)           # flattens

    assert len(lg.trades) == 1
    assert lg.trades[0]['exit_reason'] == 'stop'


def test_exit_reason_for_a_trade_still_open_at_the_end_of_the_run():
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'SA', 'SA509', 2, 100.0, 0)
    b.mark_to_market(lambda s, c: 104.0)
    lg.finish(b, datetime.date(2024, 1, 5), last_bar=4)

    assert lg.trades[0]['open_at_end'] == 1
    assert lg.trades[0]['exit_reason'] == 'end_of_run'


def test_reversal_splits_into_close_and_open_legs():
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'SA', 'SA509', 2, 100.0, 0)      # open long 2
    _do(b, lg, 'SA', 'SA509', -4, 110.0, 1)     # one order: reverses to short 2

    assert len(lg.trades) == 1
    closed = lg.trades[0]
    assert closed['direction'] == 'long'
    assert closed['close_price'] == pytest.approx(110.0)
    assert closed['gross_pnl'] == pytest.approx((110 - 100) * 2 * 20)

    assert b.net_position('SA') == -2
    assert lg._open['SA']['direction'] == 'short'
    assert lg._open['SA']['entry_qty'] == 2


def test_reconciliation_sum_net_pnl_matches_equity_change():
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'SA', 'SA509', 2, 100.0, 0)
    _do(b, lg, 'SA', 'SA509', -2, 105.0, 1, reason=Reason.ROLL)
    _do(b, lg, 'SA', 'SA601', 2, 105.0, 1, reason=Reason.ROLL)
    _do(b, lg, 'SA', 'SA601', -4, 110.0, 2)
    _do(b, lg, 'SA', 'SA601', 2, 108.0, 3)

    equity, _, _ = b.mark_to_market(lambda s, c: 108.0)
    drift = lg.reconciliation_drift(equity, 100_000.0)
    assert abs(drift) < 1e-6


def test_finish_closes_still_open_position_at_last_mark():
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'CF', 'CF509', 3, 100.0, 0)
    b.mark_to_market(lambda s, c: 106.0)   # updates pos.last_mark
    lg.finish(b, datetime.date(2024, 1, 5))

    assert len(lg.trades) == 1
    trade = lg.trades[0]
    assert trade['open_at_end'] == 1
    assert trade['close_date'] == datetime.date(2024, 1, 5)
    assert trade['gross_pnl'] == pytest.approx((106 - 100) * 3 * 5)


def test_finish_books_every_leg_a_symbol_still_holds():
    """``finish`` used to finalize on the first contract it saw and pop the
    row, so a second leg's unrealized P&L never reached the trade log at all.

    ``Engine`` keeps a product on one leg, so this is a defensive case -- but
    it is the exact shape of failure the reconciliation invariant exists to
    catch, and dropping P&L is a worse way to report a broken invariant than
    booking it.
    """
    b = Broker(100_000.0)
    lg = Ledger()

    _do(b, lg, 'SA', 'SA509', 2, 100.0, 0)
    _do(b, lg, 'SA', 'SA601', 1, 110.0, 1)      # a second leg the engine should never create
    b.positions[('SA', 'SA509')].last_mark = 105.0
    b.positions[('SA', 'SA601')].last_mark = 115.0

    lg.finish(b, D, last_bar=2)

    assert len(lg.trades) == 1
    trade = lg.trades[0]
    assert trade['open_at_end'] == 1
    assert trade['contracts'] == 'SA509|SA601'
    # both legs marked: (105-100)*2*20 + (115-110)*1*20
    assert trade['gross_pnl'] == pytest.approx(200.0 + 100.0)
    equity, _, _ = b.valuation()
    assert abs(lg.reconciliation_drift(equity, 100_000.0)) < 1e-6
