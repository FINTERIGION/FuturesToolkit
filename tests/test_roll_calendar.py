"""Roll calendar: main-month driven contract selection and roll dates."""

import pandas as pd
import pytest

from datafeed.products import roll_rule
from datafeed.roll_calendar import build_date_contract_map, roll_date, target_expiry


def _hardcoded_target(dt):
    """The calendar this module used before main_months came from the registry."""
    ts = pd.Timestamp(dt)
    year, month = ts.year, ts.month
    if month == 12:
        return year + 1, 5
    if month in (1, 2, 3):
        return year, 5
    if month in (4, 5, 6, 7):
        return year, 9
    return year + 1, 1


def test_default_rule_reproduces_the_old_hardcoded_calendar():
    for dt in pd.date_range('2014-01-01', '2030-12-31', freq='D'):
        assert target_expiry(dt) == _hardcoded_target(dt), dt


@pytest.mark.parametrize('dt, expected', [
    ('2025-03-31', (2025, 5)),    # last day on the 05 contract
    ('2025-04-01', (2025, 9)),    # rolled out of 05 on April 1st
    ('2025-07-31', (2025, 9)),
    ('2025-08-01', (2026, 1)),
    ('2025-11-30', (2026, 1)),
    ('2025-12-01', (2026, 5)),
])
def test_roll_happens_on_the_first_day_of_the_month_before_delivery(dt, expected):
    assert target_expiry(dt) == expected


def test_roll_date_is_the_month_start_one_month_before_delivery():
    assert roll_date((2025, 5)) == pd.Timestamp('2025-04-01')
    assert roll_date((2026, 1)) == pd.Timestamp('2025-12-01')
    assert roll_date((2025, 5), lead_months=2) == pd.Timestamp('2025-03-01')


@pytest.mark.parametrize('dt, expected', [
    ('2025-01-31', (2025, 3)),
    ('2025-02-01', (2025, 7)),    # 03 dropped on Feb 1st
    ('2025-09-30', (2025, 11)),
    ('2025-10-01', (2026, 3)),
])
def test_other_main_months_roll_on_their_own_schedule(dt, expected):
    assert target_expiry(dt, (3, 7, 11)) == expected


def test_single_main_month_product_rolls_once_a_year():
    assert target_expiry('2025-08-31', (10,)) == (2025, 10)
    assert target_expiry('2025-09-01', (10,)) == (2026, 10)


def test_lead_longer_than_a_year_still_finds_the_nearest_contract():
    # 15-month lead: the 05 contract is left in February of the prior year
    assert target_expiry('2025-01-31', (1, 5, 9), 15) == (2026, 5)
    assert target_expiry('2025-02-01', (1, 5, 9), 15) == (2026, 9)


@pytest.mark.parametrize('months', [(), (0, 5), (5, 13)])
def test_invalid_main_months_are_rejected(months):
    with pytest.raises(ValueError):
        target_expiry('2025-04-01', months)


def _contracts(codes, dates):
    return pd.DataFrame(
        [{'date': d, 'contract': c} for d in dates for c in codes]
    )


def test_build_map_follows_the_products_main_months():
    dates = pd.date_range('2025-03-28', '2025-04-03', freq='D')
    raw = _contracts(['SA2505', 'SA2509', 'SA2507'], dates)

    default = build_date_contract_map(dates, raw, **roll_rule('SA'))
    assert default[pd.Timestamp('2025-03-31')] == 'SA2505'
    assert default[pd.Timestamp('2025-04-01')] == 'SA2509'

    # a product registered on 05/07 would hold 05 to the same date but roll to 07
    alt = build_date_contract_map(dates, raw, main_months=(5, 7))
    assert alt[pd.Timestamp('2025-03-31')] == 'SA2505'
    assert alt[pd.Timestamp('2025-04-01')] == 'SA2507'
