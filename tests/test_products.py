"""Registry invariants and symbol-parsing round trips for datafeed.products.

Deliberately value-free: exchanges revise multipliers, margin floors, fee
schedules, and even which delivery months carry the liquidity, so pinning any
of those here would turn a routine registry edit into a test failure. What is
asserted instead is that every entry is *shaped* correctly and that the
helpers derive from it faithfully -- the parts that are this repo's own
contract rather than the exchange's.
"""

import pytest

from datafeed.products import (
    DEFAULT_MAIN_MONTHS,
    DEFAULT_ROLL_LEAD_MONTHS,
    PRODUCTS,
    SUPPORTED_EXCHANGES,
    list_products,
    parse_product,
    product_costs,
    require_products,
    roll_rule,
    tick_size,
)


@pytest.mark.parametrize('code', list(PRODUCTS))
def test_registry_entries_are_well_formed(code):
    meta = PRODUCTS[code]
    assert meta['exchange'] in SUPPORTED_EXCHANGES
    assert isinstance(meta['start_year'], int)
    assert 1990 < meta['start_year'] <= 2100
    assert meta['name'] and meta['name_zh']
    assert meta['multiplier'] > 0
    assert 0 < meta['tick_size'] <= 100      # a band, not a spec: catches a decimal slip
    assert tick_size(code) == float(meta['tick_size'])
    assert 0 < meta.get('margin_rate', 0.10) < 1
    has_rate = 'commission_rate' in meta
    has_per_lot = 'commission_per_lot' in meta
    assert has_rate != has_per_lot   # exactly one commission mode is declared

    rule = roll_rule(code)          # main months parse and are in calendar range
    assert rule['main_months']
    assert all(1 <= m <= 12 for m in rule['main_months'])
    assert rule['lead_months'] >= 1


@pytest.mark.parametrize('code', list(PRODUCTS))
def test_commission_is_a_plausible_magnitude(code):
    """A sanity band, not the schedule: catches a decimal-point slip.

    Ratio fees are quoted in 万分之N, so anything at or above 1% of notional
    is a typo; per-lot fees are single- to double-digit CNY.
    """
    costs = product_costs(code)
    if costs['commission_mode'] == 'rate':
        assert 0 < costs['commission_rate'] < 0.01
        assert costs['commission_per_lot'] is None
    else:
        assert 0 < costs['commission_per_lot'] < 1000
        assert costs['commission_rate'] is None


@pytest.mark.parametrize('code', list(PRODUCTS))
def test_product_costs_mirrors_the_registry(code):
    """``product_costs`` reports what the entry declares, whatever that is."""
    meta = PRODUCTS[code]
    costs = product_costs(code)
    assert costs['multiplier'] == meta['multiplier']
    assert costs['margin_rate'] == pytest.approx(meta.get('margin_rate', 0.10))
    if 'commission_per_lot' in meta:
        assert costs['commission_mode'] == 'per_lot'
        assert costs['commission_per_lot'] == pytest.approx(meta['commission_per_lot'])
    else:
        assert costs['commission_mode'] == 'rate'
        assert costs['commission_rate'] == pytest.approx(meta['commission_rate'])


def test_every_registered_exchange_is_supported():
    """Every declared exchange is one this toolkit can download from.

    Not the reverse: the registry is now user-editable data (products.json),
    so a supported exchange with nothing currently registered on it -- e.g.
    the last DCE product deleted through the web panel -- is not a bug.
    """
    exchanges = {PRODUCTS[c]['exchange'] for c in list_products()}
    assert exchanges <= set(SUPPORTED_EXCHANGES)


def test_list_products_preserves_definition_order():
    assert list_products() == list(PRODUCTS)


@pytest.mark.parametrize('code', list(PRODUCTS))
def test_registered_codes_round_trip_through_parse_product(code):
    assert parse_product(code) == code
    assert parse_product(code.lower()) == code
    assert parse_product(f'{code}_weighted') == code
    assert parse_product(f'{code}2601') == code       # 4-digit YYMM
    assert parse_product(f'{code}601') == code        # CZCE 3-digit YMM
    assert parse_product(f'{code}601_202601') == code  # decade-disambiguated


def test_a_prefix_code_never_swallows_a_longer_one():
    """The letter run is greedy, so ``C2601`` parses as C and ``CF2601`` as CF.

    Only meaningful while the registry holds a code that prefixes another; if
    it stops doing so there is nothing left to get wrong.
    """
    codes = list_products()
    pairs = [
        (short, long_)
        for short in codes
        for long_ in codes
        if long_ != short and long_.startswith(short)
    ]
    if not pairs:
        pytest.skip('no prefix-ambiguous codes registered')
    for short, long_ in pairs:
        assert parse_product(f'{short}2601') == short
        assert parse_product(f'{long_}2601') == long_


def test_parse_product_rejects_junk():
    for raw in (None, '', '   ', '2601', '_weighted'):
        assert parse_product(raw) is None


def test_require_products_normalizes_case_and_dedupes():
    codes = list_products()[:3]
    mixed = [c.lower() for c in codes] + [codes[0]]
    assert require_products(mixed) == codes


def test_require_products_rejects_an_unregistered_symbol():
    with pytest.raises(KeyError):
        require_products(['ZZ_NOT_REGISTERED'])


def test_roll_rule_falls_back_when_a_product_omits_the_keys():
    bare = dict(PRODUCTS[list_products()[0]])
    bare.pop('main_months', None)
    bare.pop('roll_lead_months', None)
    PRODUCTS['ZZ_TEST'] = bare
    try:
        assert roll_rule('ZZ_TEST') == {
            'main_months': DEFAULT_MAIN_MONTHS,
            'lead_months': DEFAULT_ROLL_LEAD_MONTHS,
        }
    finally:
        del PRODUCTS['ZZ_TEST']


def test_roll_rule_honours_a_declared_override():
    """A product off the default cycle -- SHFE rebar, silver -- is expressed
    by declaring the keys, so the override path is what matters, not which
    months any particular product happens to use today."""
    override = dict(PRODUCTS[list_products()[0]])
    override['main_months'] = (12, 6, 6, 9)     # unsorted + duplicated on purpose
    override['roll_lead_months'] = 2
    PRODUCTS['ZZ_TEST'] = override
    try:
        assert roll_rule('ZZ_TEST') == {'main_months': (6, 9, 12), 'lead_months': 2}
    finally:
        del PRODUCTS['ZZ_TEST']


def test_roll_rule_rejects_an_out_of_range_month():
    broken = dict(PRODUCTS[list_products()[0]])
    broken['main_months'] = (1, 13)
    PRODUCTS['ZZ_TEST'] = broken
    try:
        with pytest.raises(ValueError):
            roll_rule('ZZ_TEST')
    finally:
        del PRODUCTS['ZZ_TEST']


def test_parse_product_rejects_a_contract_code_for_an_unregistered_product():
    """A string shaped like a contract code is not a product unless the
    registry carries it. Returning the letter run regardless produced a
    plausible code that every costing and rolling helper then rejected, so the
    failure surfaced as a KeyError deep inside `product_costs` rather than
    where the unknown string came in."""
    unregistered = 'ZZ'
    assert unregistered not in PRODUCTS          # guard the premise
    for raw in (unregistered, f'{unregistered}2601', f'{unregistered}601',
                f'{unregistered}601_202601', f'{unregistered}_weighted'):
        assert parse_product(raw) is None, raw


def test_parse_product_result_is_always_costable():
    """The contract `parse_product` now keeps: a non-None answer names a
    product the rest of the registry can price and roll."""
    for code in list(PRODUCTS):
        for raw in (code, code.lower(), f'{code}_weighted', f'{code}2601', f'{code}601'):
            parsed = parse_product(raw)
            assert parsed is not None
            product_costs(parsed)                # would raise KeyError if not registered
            roll_rule(parsed)
