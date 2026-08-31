"""Registry invariants and symbol-parsing round trips for datafeed.products."""

import pytest

from datafeed.products import (
    PRODUCTS,
    list_products,
    parse_product,
    product_costs,
    require_products,
    roll_rule,
)


@pytest.mark.parametrize('code', list(PRODUCTS))
def test_registry_entries_are_well_formed(code):
    meta = PRODUCTS[code]
    assert meta['exchange'] == 'CZCE'
    assert isinstance(meta['start_year'], int)
    assert meta['multiplier'] > 0
    assert 0 < meta.get('margin_rate', 0.10) < 1
    has_rate = 'commission_rate' in meta
    has_per_lot = 'commission_per_lot' in meta
    assert has_rate != has_per_lot   # exactly one commission mode is declared

    rule = roll_rule(code)          # main months parse and are in calendar range
    assert rule['main_months']
    assert all(1 <= m <= 12 for m in rule['main_months'])
    assert rule['lead_months'] >= 1


def test_list_products_includes_all_seven():
    assert list_products() == ['SA', 'FG', 'CF', 'MA', 'TA', 'SR', 'OI']


@pytest.mark.parametrize('code, mode, value', [
    ('MA', 'rate', 0.0001),
    ('TA', 'per_lot', 3.0),
    ('SR', 'per_lot', 3.0),
    ('OI', 'per_lot', 2.0),
])
def test_new_products_cost_mode(code, mode, value):
    costs = product_costs(code)
    assert costs['commission_mode'] == mode
    assert costs[f'commission_{mode}'] == value
    assert costs['multiplier'] == PRODUCTS[code]['multiplier']


@pytest.mark.parametrize('raw, expected', [
    ('MA509', 'MA'),
    ('TA2509', 'TA'),
    ('OI_weighted', 'OI'),
    ('SR501_202501', 'SR'),
    ('sr', 'SR'),
])
def test_parse_product_round_trip(raw, expected):
    assert parse_product(raw) == expected


def test_require_products_accepts_new_symbols():
    assert require_products(['ma', 'TA', 'sr', 'OI']) == ['MA', 'TA', 'SR', 'OI']


def test_registered_products_roll_on_the_czce_01_05_09_cycle():
    for code in list_products():
        assert roll_rule(code) == {'main_months': (1, 5, 9), 'lead_months': 1}


def test_roll_rule_falls_back_when_a_product_omits_the_keys():
    bare = dict(PRODUCTS['SA'])
    bare.pop('main_months', None)
    PRODUCTS['ZZ_TEST'] = bare
    try:
        assert roll_rule('ZZ_TEST') == {'main_months': (1, 5, 9), 'lead_months': 1}
    finally:
        del PRODUCTS['ZZ_TEST']
