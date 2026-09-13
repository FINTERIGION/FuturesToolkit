import datetime

import pytest

from core.broker import Broker
from datafeed.products import product_costs

D = datetime.date(2024, 1, 1)
CF = product_costs('CF')


def test_realized_pnl_on_full_close_long():
    b = Broker(100_000.0)
    b.fill('CF', 'CF509', 2, 100.0, 0, D)          # CF: multiplier=5, commission_per_lot=4.3
    f = b.fill('CF', 'CF509', -2, 110.0, 1, D)
    assert f.realized_pnl == pytest.approx((110 - 100) * 2 * 5)
    assert ('CF', 'CF509') not in b.positions


def test_commission_per_lot_mode():
    b = Broker(100_000.0)
    f = b.fill('CF', 'CF509', 1, 1000.0, 0, D)
    assert f.commission == pytest.approx(4.3)


def test_commission_rate_mode():
    b = Broker(100_000.0)
    f = b.fill('SA', 'SA509', 1, 1000.0, 0, D)     # SA: multiplier=20, commission_rate=0.0001
    assert f.commission == pytest.approx(1000 * 20 * 0.0001)


def test_long_short_symmetry():
    """The bug this design fixes: backtrader's margin-in-equity leak made
    longs and shorts report different-magnitude P&L for a symmetric move."""
    b1 = Broker(100_000.0)
    b1.fill('CF', 'CF509', 2, 100.0, 0, D)
    b1.fill('CF', 'CF509', -2, 110.0, 1, D)
    eq1, _, _ = b1.mark_to_market(lambda s, c: 110.0)

    b2 = Broker(100_000.0)
    b2.fill('CF', 'CF509', -2, 100.0, 0, D)
    b2.fill('CF', 'CF509', 2, 90.0, 1, D)
    eq2, _, _ = b2.mark_to_market(lambda s, c: 90.0)

    assert (eq1 - 100_000.0) == pytest.approx(eq2 - 100_000.0)


def test_mark_to_market_includes_unrealized_and_margin():
    b = Broker(100_000.0)
    f = b.fill('CF', 'CF509', 1, 100.0, 0, D)
    eq, margin_used, available = b.mark_to_market(lambda s, c: 105.0)
    expected_equity = 100_000.0 - f.commission + (105 - 100) * 1 * 5
    assert eq == pytest.approx(expected_equity)
    # the registry owns the rate; this asserts the broker's arithmetic on it
    assert margin_used == pytest.approx(
        1 * 105 * CF['multiplier'] * CF['margin_rate']
    )
    assert available == pytest.approx(eq - margin_used)


def test_dark_bar_carries_forward_last_mark():
    b = Broker(100_000.0)
    b.fill('CF', 'CF509', 1, 100.0, 0, D)
    eq1, _, _ = b.mark_to_market(lambda s, c: 108.0)
    eq2, _, _ = b.mark_to_market(lambda s, c: None)   # no print today
    assert eq2 == pytest.approx(eq1)


def test_force_liquidation_stops_further_exposure_without_spiraling():
    b = Broker(1_000.0)
    b.fill('SA', 'SA509', 3, 100.0, 0, D)             # notional 6000, margin 720 at entry
    equity_before, margin_used, available = b.mark_to_market(lambda s, c: 90.0)
    assert available < 0
    assert equity_before > 0   # margin call triggers well before equity itself goes negative

    fills = b.force_liquidate(1, D)
    assert len(fills) == 1
    assert b.positions == {}

    equity_after, margin_after, _ = b.mark_to_market(lambda s, c: 90.0)
    assert margin_after == 0.0
    assert equity_after == pytest.approx(equity_before - fills[0].commission)

    # Flat: no further exposure to subsequent price moves.
    equity_later, _, _ = b.mark_to_market(lambda s, c: 10.0)
    assert equity_later == pytest.approx(equity_after)


def test_force_liquidation_prices_each_fill_through_price_for():
    """The broker knows nothing about ticks; the engine hands it the slippage."""
    b = Broker(1_000.0)
    b.fill('SA', 'SA509', 3, 100.0, 0, D)
    b.mark_to_market(lambda s, c: 90.0)

    seen = []

    def price_for(symbol, size, mark):
        seen.append((symbol, size, mark))
        return mark - 2.0

    fills = b.force_liquidate(1, D, price_for=price_for)
    assert seen == [('SA', -3, 90.0)]
    assert fills[0].price == pytest.approx(88.0)
    assert b.positions == {}


# --------------------------------------------------------------------------
# Pre-trade margin check
# --------------------------------------------------------------------------

def test_an_opening_order_that_does_not_fit_is_refused():
    """CF at 15,000 needs 15,000 x 5 x 0.10 = 7,500 margin per lot."""
    b = Broker(10_000.0)
    assert b.can_afford('CF', 'CF509', 1, 15_000.0)      # 7,500 of 10,000
    assert not b.can_afford('CF', 'CF509', 2, 15_000.0)  # 15,000 of 10,000


def test_adding_to_a_position_is_measured_against_the_whole_book():
    b = Broker(10_000.0)
    b.fill('CF', 'CF509', 1, 15_000.0, 0, D)
    b.mark_to_market(lambda s, c: 15_000.0)
    assert not b.can_afford('CF', 'CF509', 1, 15_000.0)  # the 2nd lot no longer fits


def test_a_closing_order_is_never_refused_even_inside_a_margin_call():
    """The exemption that keeps a rejection from trapping a position.

    Marked down to where equity is far below the margin requirement, an
    opening order is refused but the exit still has to go through.
    """
    b = Broker(10_000.0)
    b.fill('CF', 'CF509', 1, 15_000.0, 0, D)
    b.mark_to_market(lambda s, c: 13_500.0)              # equity 2,500, margin 6,750
    equity, margin_used, available = b.valuation()
    assert available < 0                                  # in a margin call

    assert not b.can_afford('CF', 'CF509', 1, 13_500.0)   # cannot add
    assert b.can_afford('CF', 'CF509', -1, 13_500.0)      # can always get out


def test_a_flip_is_judged_on_the_margin_it_would_leave_behind():
    """+1 -> -1 is one order for 2 lots but ends at the same 1-lot margin."""
    b = Broker(8_000.0)
    b.fill('CF', 'CF509', 1, 15_000.0, 0, D)
    b.mark_to_market(lambda s, c: 15_000.0)
    assert b.can_afford('CF', 'CF509', -2, 15_000.0)      # ends flat-then-short, 7,500
    assert not b.can_afford('CF', 'CF509', -3, 15_000.0)  # ends 2 lots short, 15,000


def test_valuation_does_not_move_a_position_mark():
    b = Broker(100_000.0)
    b.fill('CF', 'CF509', 1, 15_000.0, 0, D)
    b.mark_to_market(lambda s, c: 16_000.0)
    before = b.positions[('CF', 'CF509')].last_mark
    equity, margin_used, available = b.valuation()
    assert b.positions[('CF', 'CF509')].last_mark == before
    assert equity == pytest.approx(100_000.0 + (16_000 - 15_000) * 5 - 4.3)
    assert margin_used == pytest.approx(16_000 * 5 * 0.10)


def test_adding_to_a_position_marks_it_at_the_price_just_paid():
    """``valuation()`` is what the OPEN-phase margin check reads, and it runs
    mid-bar -- after some symbols have filled, before today's settle exists.
    The add branch was the one fill that left ``last_mark`` alone, so every
    symbol behind it in that loop sized its margin off yesterday's price.
    """
    b = Broker(1_000_000.0)
    b.fill('SA', 'SA509', 2, 100.0, 0, D)
    b.positions[('SA', 'SA509')].last_mark = 100.0      # as SETTLE would leave it
    b.fill('SA', 'SA509', 2, 130.0, 1, D)               # add, at a much higher open

    pos = b.positions[('SA', 'SA509')]
    assert pos.size == 4
    assert pos.avg_entry == pytest.approx(115.0)
    assert pos.last_mark == pytest.approx(130.0)

    # margin now reflects today's price, not yesterday's
    _, margin_used, _ = b.valuation()
    assert margin_used == pytest.approx(4 * 130.0 * 20 * 0.11)


def test_every_fill_leaves_a_usable_mark():
    """Open, add, trim and flip: whichever branch ran, the position carries the
    price it last actually traded at."""
    b = Broker(1_000_000.0)
    for size, price in ((2, 100.0), (2, 110.0), (-1, 120.0), (-5, 130.0)):
        b.fill('SA', 'SA509', size, price, 0, D)
        pos = b.positions.get(('SA', 'SA509'))
        if pos is not None:
            assert pos.last_mark == pytest.approx(price)
