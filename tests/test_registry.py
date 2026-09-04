"""Load/validate/save/reload round trip for the JSON product registry.

``tests/test_products.py`` covers the shape of whatever is currently on
disk; this covers the loader and writer machinery itself -- the part that
becomes live once products are added, edited, and deleted through the web
panel rather than by hand in a Python literal.
"""

from __future__ import annotations

import json

import pytest

from datafeed.products import (
    PRODUCTS,
    _load_registry,
    save_registry,
    validate_product,
)


@pytest.fixture(autouse=True)
def _restore_products():
    """``save_registry`` intentionally mutates the shared, process-global
    ``PRODUCTS`` dict in place -- regardless of which ``path`` it wrote to --
    so every module already holding a reference sees fresh data without
    re-importing. That means any test here calling it would otherwise leak
    the mutation into whichever test runs next; restore the exact in-memory
    snapshot afterward rather than re-reading the real file, so this has no
    dependency on the real registry's content or on test order.
    """
    snapshot = {code: dict(meta) for code, meta in PRODUCTS.items()}
    order = list(PRODUCTS)
    yield
    PRODUCTS.clear()
    PRODUCTS.update({code: snapshot[code] for code in order})


_VALID = {
    'exchange': 'CZCE',
    'name': 'test_product',
    'name_zh': '测试品种',
    'start_year': 2020,
    'main_months': [1, 5, 9],
    'multiplier': 10,
    'tick_size': 1.0,
    'margin_rate': 0.12,
    'commission_rate': 0.0001,
}


def test_validate_product_accepts_a_well_formed_entry():
    out = validate_product('zz', _VALID)
    assert out['main_months'] == (1, 5, 9)          # normalized to a tuple
    assert out['margin_rate'] == pytest.approx(0.12)
    assert 'commission_per_lot' not in out


@pytest.mark.parametrize('bad_code', ['', '  ', 'ZZ-1', 'ZZ 1', 'ZZ.1'])
def test_validate_product_rejects_a_malformed_code(bad_code):
    with pytest.raises(ValueError):
        validate_product(bad_code, _VALID)


def test_validate_product_rejects_unsupported_exchange():
    bad = dict(_VALID, exchange='NYSE')
    with pytest.raises(ValueError, match='exchange'):
        validate_product('ZZ', bad)


@pytest.mark.parametrize('bad_year', [1800, 2200, 2020.5, '2020'])
def test_validate_product_rejects_bad_start_year(bad_year):
    bad = dict(_VALID, start_year=bad_year)
    with pytest.raises(ValueError, match='start_year'):
        validate_product('ZZ', bad)


@pytest.mark.parametrize('field', ['name', 'name_zh'])
def test_validate_product_rejects_empty_names(field):
    bad = dict(_VALID, **{field: ''})
    with pytest.raises(ValueError, match=field):
        validate_product('ZZ', bad)


@pytest.mark.parametrize('bad_mult', [0, -5])
def test_validate_product_rejects_non_positive_multiplier(bad_mult):
    bad = dict(_VALID, multiplier=bad_mult)
    with pytest.raises(ValueError, match='multiplier'):
        validate_product('ZZ', bad)


@pytest.mark.parametrize('bad_tick', [0, -1, 150])
def test_validate_product_rejects_tick_size_out_of_band(bad_tick):
    bad = dict(_VALID, tick_size=bad_tick)
    with pytest.raises(ValueError, match='tick_size'):
        validate_product('ZZ', bad)


@pytest.mark.parametrize('bad_margin', [0, 1, -0.1, 1.5])
def test_validate_product_rejects_margin_rate_out_of_band(bad_margin):
    bad = dict(_VALID, margin_rate=bad_margin)
    with pytest.raises(ValueError, match='margin_rate'):
        validate_product('ZZ', bad)


def test_validate_product_rejects_neither_commission_mode():
    bad = dict(_VALID)
    bad.pop('commission_rate')
    with pytest.raises(ValueError, match='commission'):
        validate_product('ZZ', bad)


def test_validate_product_rejects_both_commission_modes():
    bad = dict(_VALID, commission_per_lot=3.0)  # already has commission_rate
    with pytest.raises(ValueError, match='commission'):
        validate_product('ZZ', bad)


def test_validate_product_accepts_commission_per_lot():
    good = dict(_VALID)
    good.pop('commission_rate')
    good['commission_per_lot'] = 3.5
    out = validate_product('ZZ', good)
    assert out['commission_per_lot'] == pytest.approx(3.5)
    assert 'commission_rate' not in out


def test_validate_product_rejects_out_of_range_main_months():
    bad = dict(_VALID, main_months=[1, 13])
    with pytest.raises(ValueError):
        validate_product('ZZ', bad)


def test_validate_product_rejects_bad_roll_lead_months():
    bad = dict(_VALID, roll_lead_months=0)
    with pytest.raises(ValueError, match='roll_lead_months'):
        validate_product('ZZ', bad)


def test_validate_product_lowercases_and_normalizes_code():
    out = validate_product('zz', _VALID)
    assert out['multiplier'] == 10


def test_load_registry_preserves_order_and_tuple_main_months(tmp_path):
    path = tmp_path / 'products.json'
    path.write_text(json.dumps({
        'BB': dict(_VALID),
        'AA': dict(_VALID),
    }, ensure_ascii=False), encoding='utf-8')
    loaded = _load_registry(str(path))
    assert list(loaded) == ['BB', 'AA']
    assert loaded['BB']['main_months'] == (1, 5, 9)


def test_save_registry_round_trips_through_disk(tmp_path):
    path = tmp_path / 'products.json'
    save_registry({'ZZ': _VALID}, path=str(path))
    assert path.exists()

    on_disk = json.loads(path.read_text(encoding='utf-8'))
    assert on_disk['ZZ']['main_months'] == [1, 5, 9]   # JSON array, not a tuple

    reloaded = _load_registry(str(path))
    assert reloaded['ZZ']['main_months'] == (1, 5, 9)


def test_save_registry_rejects_an_invalid_entry_and_writes_nothing(tmp_path):
    path = tmp_path / 'products.json'
    original = {'ZZ': _VALID}
    save_registry(original, path=str(path))
    before = path.read_text(encoding='utf-8')

    broken = {'ZZ': dict(_VALID), 'YY': dict(_VALID, multiplier=-1)}
    with pytest.raises(ValueError):
        save_registry(broken, path=str(path))

    after = path.read_text(encoding='utf-8')
    assert after == before                          # untouched: no partial write
    assert not list(path.parent.glob('.products.*.tmp'))  # no leftover tmp file


def test_save_registry_updates_PRODUCTS_in_place(tmp_path):
    path = tmp_path / 'products.json'
    target = PRODUCTS
    save_registry({'ZZTEST': _VALID}, path=str(path))
    assert PRODUCTS is target                        # same dict object
    assert list(PRODUCTS) == ['ZZTEST']               # new content
