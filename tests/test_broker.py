import datetime

import pytest

from core.broker import Broker

D = datetime.date(2024, 1, 1)


def test_realized_pnl_on_full_close_long():
    b = Broker(100_000.0)
    b.fill('FG', 'FG509', 2, 100.0, 0, D)          # FG: multiplier=20, commission_per_lot=2.0
    f = b.fill('FG', 'FG509', -2, 110.0, 1, D)
    assert f.realized_pnl == pytest.approx((110 - 100) * 2 * 20)
    assert ('FG', 'FG509') not in b.positions


def test_commission_per_lot_mode():
    b = Broker(100_000.0)
    f = b.fill('FG', 'FG509', 1, 1000.0, 0, D)
    assert f.commission == pytest.approx(2.0)


def test_commission_rate_mode():
    b = Broker(100_000.0)
    f = b.fill('SA', 'SA509', 1, 1000.0, 0, D)     # SA: multiplier=20, commission_rate=0.0001
    assert f.commission == pytest.approx(1000 * 20 * 0.0001)


def test_long_short_symmetry():
    """The bug this design fixes: backtrader's margin-in-equity leak made
    longs and shorts report different-magnitude P&L for a symmetric move."""
    b1 = Broker(100_000.0)
    b1.fill('FG', 'FG509', 2, 100.0, 0, D)
    b1.fill('FG', 'FG509', -2, 110.0, 1, D)
    eq1, _, _ = b1.mark_to_market(lambda s, c: 110.0)

    b2 = Broker(100_000.0)
    b2.fill('FG', 'FG509', -2, 100.0, 0, D)
    b2.fill('FG', 'FG509', 2, 90.0, 1, D)
    eq2, _, _ = b2.mark_to_market(lambda s, c: 90.0)

    assert (eq1 - 100_000.0) == pytest.approx(eq2 - 100_000.0)


def test_mark_to_market_includes_unrealized_and_margin():
    b = Broker(100_000.0)
    f = b.fill('FG', 'FG509', 1, 100.0, 0, D)
    eq, margin_used, available = b.mark_to_market(lambda s, c: 105.0)
    expected_equity = 100_000.0 - f.commission + (105 - 100) * 1 * 20
    assert eq == pytest.approx(expected_equity)
    assert margin_used == pytest.approx(1 * 105 * 20 * 0.13)   # FG margin_rate=0.13
    assert available == pytest.approx(eq - margin_used)


def test_dark_bar_carries_forward_last_mark():
    b = Broker(100_000.0)
    b.fill('FG', 'FG509', 1, 100.0, 0, D)
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
