"""Parsing, calendar bookkeeping, and auth retries for datafeed.sources.

Everything here runs offline against inline fixtures -- no exchange is contacted,
matching the "synthetic data, no real CSVs" convention in conftest.py.
"""

import gzip
import json
import os
import threading
import urllib.error
from datetime import date, timedelta

import pandas as pd
import pytest

from datafeed import sources
from datafeed.products import PRODUCTS, parse_product
from datafeed.roll_calendar import parse_contract_expiry
from datafeed.sources import (
    NoDataForDate,
    _clean_contract_bars,
    _DailyFileSource,
    _DceAuth,
    _DceSource,
    _ShfeSource,
    _text,
    get_source,
)

TODAY = date.today()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def shfe_payload():
    """One SHFE day: two rb contracts, a subtotal, a foreign product, a total."""
    return {
        'report_date': '20260828',
        'o_curinstrument': [
            {'PRODUCTID': 'rb_f', 'DELIVERYMONTH': '2610', 'OPENPRICE': 3100,
             'HIGHESTPRICE': 3150, 'LOWESTPRICE': 3080, 'CLOSEPRICE': 3120,
             'SETTLEMENTPRICE': 3110, 'VOLUME': 250000, 'OPENINTEREST': 1800000},
            # listed but untraded: the exchange sends empty strings, not nulls
            {'PRODUCTID': 'rb_f', 'DELIVERYMONTH': '2707', 'OPENPRICE': '',
             'HIGHESTPRICE': '', 'LOWESTPRICE': '', 'CLOSEPRICE': '',
             'SETTLEMENTPRICE': 3115, 'VOLUME': 0, 'OPENINTEREST': 12},
            {'PRODUCTID': 'rb_f', 'DELIVERYMONTH': '小计', 'OPENPRICE': '',
             'VOLUME': 1554680, 'OPENINTEREST': 2860293},
            {'PRODUCTID': 'hc_f', 'DELIVERYMONTH': '2610', 'OPENPRICE': 3400,
             'HIGHESTPRICE': 3420, 'LOWESTPRICE': 3390, 'CLOSEPRICE': 3410,
             'SETTLEMENTPRICE': 3405, 'VOLUME': 90000, 'OPENINTEREST': 700000},
            {'PRODUCTID': 'auefp', 'DELIVERYMONTH': '2610', 'OPENPRICE': 1,
             'CLOSEPRICE': 1, 'VOLUME': 1, 'OPENINTEREST': 1},
            {'PRODUCTID': '总计', 'DELIVERYMONTH': '', 'VOLUME': 9, 'OPENINTEREST': 9},
        ],
    }


def dce_payload():
    """One DCE day: corn beside corn starch, coking coal beside coke, an option."""
    def row(cid, **kw):
        base = {'contractId': cid, 'variety': 'x', 'open': '1', 'high': '2',
                'low': '3', 'close': '4', 'clearPrice': '5', 'volumn': 6,
                'openInterest': 7}
        base.update(kw)
        return base

    return {
        'success': True,
        'code': 200,
        'data': [
            row('c2601', open='2251', high='2271', low='2245', close='2255',
                clearPrice='2257', volumn=2931, openInterest=3753),
            row('c2603'),
            row('cs2601'),          # corn starch -- must not land in C
            row('jm2601'),
            row('j2601'),           # coke -- must not land in JM
            row('v2601'),
            row('c2607-C-2040'),    # option -- must not land anywhere
            {'variety': '总计', 'contractId': None, 'open': None, 'close': None,
             'volumn': 0, 'openInterest': 0},
        ],
    }


def make_source(cls, symbol, cache_dir, start_year=2015):
    meta = dict(PRODUCTS[symbol], start_year=start_year)
    return cls(symbol, meta, str(cache_dir))


# --------------------------------------------------------------------------
# SHFE extraction
# --------------------------------------------------------------------------

def test_shfe_keeps_only_the_requested_product(tmp_path):
    src = make_source(_ShfeSource, 'RB', tmp_path)
    rows = src._extract(shfe_payload(), date(2026, 8, 28))
    assert [r['contract'] for r in rows] == ['RB2610', 'RB2707']


def test_shfe_drops_subtotal_and_total_rows(tmp_path):
    src = make_source(_ShfeSource, 'RB', tmp_path)
    rows = src._extract(shfe_payload(), date(2026, 8, 28))
    assert all(r['contract'][2:].isdigit() for r in rows)
    assert not any('小计' in r['contract'] or '总计' in r['contract'] for r in rows)


def test_shfe_maps_the_price_fields(tmp_path):
    src = make_source(_ShfeSource, 'RB', tmp_path)
    row = src._extract(shfe_payload(), date(2026, 8, 28))[0]
    assert row == {
        'date': '2026-08-28', 'contract': 'RB2610', 'open': '3100',
        'high': '3150', 'low': '3080', 'close': '3120', 'settle': '3110',
        'oi': '1800000', 'volume': '250000',
    }


def test_shfe_empty_instrument_list_is_a_non_trading_day(tmp_path, monkeypatch):
    """New Year's Day answers 200 with a 31-byte body, not 404.

    Treating that as a short/failed transfer would leave the day unclassified
    and re-requested on every future sync.
    """
    body = b'{    "o_curinstrument": []}'
    monkeypatch.setattr(sources, '_http_bytes', lambda *a, **k: body)
    src = make_source(_ShfeSource, 'RB', tmp_path)
    with pytest.raises(NoDataForDate):
        src._fetch_day(date(2016, 1, 1))


def test_shfe_trading_day_is_not_mistaken_for_a_holiday(tmp_path, monkeypatch):
    payload = json.dumps(shfe_payload()).encode('utf-8')
    monkeypatch.setattr(sources, '_http_bytes', lambda *a, **k: payload)
    src = make_source(_ShfeSource, 'RB', tmp_path)
    assert src._fetch_day(date(2026, 8, 28)) == payload


def test_shfe_empty_prices_survive_as_nan(tmp_path):
    src = make_source(_ShfeSource, 'RB', tmp_path)
    rows = src._extract(shfe_payload(), date(2026, 8, 28))
    frame = _clean_contract_bars(pd.DataFrame(rows, columns=sources.REQUIRED_COLUMNS))
    untraded = frame[frame['contract'] == 'RB2707'].iloc[0]
    assert pd.isna(untraded['close'])
    assert untraded['oi'] == 12


# --------------------------------------------------------------------------
# DCE extraction
# --------------------------------------------------------------------------

@pytest.mark.parametrize('symbol, expected', [
    ('C', ['C2601', 'C2603']),      # not cs2601, not the c2607-C-2040 option
    ('JM', ['JM2601']),             # not j2601
])
def test_dce_contract_regex_is_anchored(tmp_path, symbol, expected):
    src = make_source(_DceSource, symbol, tmp_path)
    rows = src._extract(dce_payload(), date(2026, 1, 5))
    assert [r['contract'] for r in rows] == expected


def test_dce_maps_clear_price_and_the_misspelled_volume(tmp_path):
    src = make_source(_DceSource, 'C', tmp_path)
    row = src._extract(dce_payload(), date(2026, 1, 5))[0]
    assert row == {
        'date': '2026-01-05', 'contract': 'C2601', 'open': '2251',
        'high': '2271', 'low': '2245', 'close': '2255', 'settle': '2257',
        'oi': '3753', 'volume': '2931',
    }


def test_dce_summary_only_payload_is_a_non_trading_day(tmp_path, monkeypatch):
    """A holiday answers 200 with a lone 总计 row, not an empty array."""
    src = make_source(_DceSource, 'C', tmp_path)
    holiday = {'success': True, 'code': 200, 'data': [
        {'variety': '总计', 'contractId': None, 'volumn': 0, 'openInterest': 0},
    ]}
    monkeypatch.setattr(src, '_call', lambda *a, **k: holiday)
    with pytest.raises(NoDataForDate):
        src._fetch_day(date(2015, 1, 3))


def test_dce_trading_day_is_not_mistaken_for_a_holiday(tmp_path, monkeypatch):
    src = make_source(_DceSource, 'C', tmp_path)
    monkeypatch.setattr(src, '_call', lambda *a, **k: dce_payload())
    assert src._fetch_day(date(2026, 1, 5))


# --------------------------------------------------------------------------
# Contract codes round-trip into the roll calendar
# --------------------------------------------------------------------------

@pytest.mark.parametrize('code, expiry', [
    ('RB1505', (2015, 5)),      # 4-digit YYMM in 2015, not 2005
    ('RB2612', (2026, 12)),
    ('C2601', (2026, 1)),
    ('JM2605', (2026, 5)),
])
def test_codes_parse_into_the_roll_calendar(code, expiry):
    assert parse_contract_expiry(code, '2015-01-05') == expiry
    assert parse_product(code) == ''.join(c for c in code if c.isalpha())


# --------------------------------------------------------------------------
# The string contract between sources and _clean_contract_bars
# --------------------------------------------------------------------------

def test_text_renders_scalars_as_strings():
    assert _text(None) == ''
    assert _text(0) == '0'
    assert _text('') == ''
    assert _text(3.5) == '3.5'


def test_mixed_int_and_blank_column_would_be_wiped_without_text():
    """Guards the reason _text exists.

    SHFE sends a traded contract's price as a JSON int and an untraded one's as
    ``''``. Those land in one object-dtype column, and ``_clean_contract_bars``
    strips object columns with ``.str.strip()`` -- an accessor that turns every
    non-string element into NaN. Stringifying first is what keeps the 3100.
    """
    rows = [
        {'date': '2026-01-05', 'contract': 'RB2610', 'open': 3100, 'high': 3100,
         'low': 3100, 'close': 3100, 'settle': 3100, 'oi': 1, 'volume': 1},
        {'date': '2026-01-05', 'contract': 'RB2707', 'open': '', 'high': '',
         'low': '', 'close': '', 'settle': '', 'oi': 1, 'volume': 1},
    ]
    raw = pd.DataFrame(rows, columns=sources.REQUIRED_COLUMNS)
    assert raw['open'].dtype == object                       # int next to ''
    assert pd.isna(_clean_contract_bars(raw)['open'].iloc[0])

    stringified = pd.DataFrame(
        [{k: _text(v) for k, v in row.items()} for row in rows],
        columns=sources.REQUIRED_COLUMNS,
    )
    assert _clean_contract_bars(stringified)['open'].iloc[0] == 3100


# --------------------------------------------------------------------------
# Trading-calendar bookkeeping
# --------------------------------------------------------------------------

class _FakeDaily(_DailyFileSource):
    exchange = 'FAKE'
    prefix = 'fake'
    throttle = 0.0

    def __init__(self, symbol, meta, cache_dir, available=(), broken=(), version='a'):
        super().__init__(symbol, meta, cache_dir)
        self.available = set(available)
        self.broken = set(broken)
        self.version = version
        self.requested = []

    def _fetch_day(self, day):
        key = day.isoformat()
        self.requested.append(key)
        if key in self.broken:
            raise RuntimeError('transient network failure')
        if key not in self.available:
            raise NoDataForDate(key)
        return json.dumps(
            {'v': self.version, 'rows': [{'contract': f'X{key[2:4]}01'}]}
        ).encode('utf-8')

    def _extract(self, payload, day):
        return [{'date': day.strftime('%Y-%m-%d'), 'contract': r['contract'],
                 'open': '1', 'high': '1', 'low': '1', 'close': '1',
                 'settle': '1', 'oi': '1', 'volume': '1'}
                for r in payload['rows']]


def _early_weekdays(n):
    """The first ``n`` weekdays of the current year -- safely outside the tail."""
    out, day = [], date(TODAY.year, 1, 1)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _fake(tmp_path, available, broken=(), version='a'):
    meta = dict(PRODUCTS['RB'], start_year=TODAY.year)
    return _FakeDaily('RB', meta, str(tmp_path), available=available,
                      broken=broken, version=version)


def _recent_weekday():
    """The latest weekday, which is always inside the re-fetch tail."""
    day = TODAY
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def test_holidays_are_remembered_and_failures_are_not(tmp_path):
    traded, holiday, flaky = _early_weekdays(3)
    src = _fake(tmp_path, available=[traded.isoformat()], broken=[flaky.isoformat()])
    src.sync()

    state = src._read_calendar()
    assert holiday.isoformat() in state['no_data']
    # a transient failure must stay retryable, not be cached as "no session"
    assert flaky.isoformat() not in state['no_data']
    assert state['last_checked'] == TODAY.isoformat()
    assert src.cache_path(traded).endswith(f'fake{traded:%Y%m%d}.json.gz')


def test_second_sync_reuses_cache_but_retries_failures(tmp_path):
    traded, holiday, flaky = _early_weekdays(3)
    src = _fake(tmp_path, available=[traded.isoformat()], broken=[flaky.isoformat()])
    src.sync()

    again = _fake(tmp_path, available=[traded.isoformat()], broken=[flaky.isoformat()])
    again.sync()
    assert traded.isoformat() not in again.requested      # cache hit
    assert holiday.isoformat() not in again.requested     # known holiday
    assert flaky.isoformat() in again.requested           # still worth a retry


def test_unchanged_tail_refetch_is_not_reported_as_refreshed(tmp_path):
    """The tail is always re-fetched; only moved bytes should force a rebuild."""
    recent = _recent_weekday()
    src = _fake(tmp_path, available=[recent.isoformat()])
    assert recent.isoformat() in src.sync()

    again = _fake(tmp_path, available=[recent.isoformat()])
    refreshed = again.sync()
    assert recent.isoformat() in again.requested      # re-fetched, as designed
    assert recent.isoformat() not in refreshed        # but nothing changed


def test_revised_tail_day_is_reported_as_refreshed(tmp_path):
    recent = _recent_weekday()
    _fake(tmp_path, available=[recent.isoformat()], version='a').sync()

    revised = _fake(tmp_path, available=[recent.isoformat()], version='b')
    assert recent.isoformat() in revised.sync()
    assert revised._cached_payload(recent) is not None
    assert b'"v": "b"' in revised._cached_payload(recent)


def test_concurrent_writers_cannot_corrupt_a_shared_day_cache(tmp_path):
    """Two products on one venue may be updated at the same time.

    A day payload is shared by every product on the exchange, so both writers
    resolve the same ``cache_path``. The web panel runs two update jobs at once
    and de-duplicates only by product (``web/routers/data.py``), which puts
    this one extra click away rather than in the realm of theory. With the
    fixed ``f'{path}.tmp'`` this used to be, both writers opened the *same*
    temp file: the loser either spliced its tail into the winner's
    already-renamed inode or died on ``FileNotFoundError`` partway through its
    own job. Eight writers here, because the failure is a race and one pass
    proves nothing.
    """
    src = _fake(tmp_path, available=[])
    path = src.cache_path(_recent_weekday())
    errors = []
    barrier = threading.Barrier(8)

    def writer(tag: bytes):
        try:
            barrier.wait()
            for _ in range(15):
                src._write_day(path, tag * 5000)
        except Exception as exc:      # noqa: BLE001 -- that there are none is the assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(bytes([c]),))
               for c in b'ABCDEFGH']
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    with gzip.open(path, 'rb') as fh:
        payload = fh.read()
    # Exactly one writer's bytes, whole -- not a splice of several.
    assert len(payload) == 5000
    assert len(set(payload)) == 1
    # And nothing left over for `_cached_days` to walk past on the next sync.
    assert os.listdir(src.dir) == [os.path.basename(path)]


def test_weekends_are_never_requested(tmp_path):
    src = _fake(tmp_path, available=[])
    src.sync()
    assert all(date.fromisoformat(k).weekday() < 5 for k in src.requested)


def test_load_bars_reads_every_cached_day(tmp_path):
    days = _early_weekdays(4)
    src = _fake(tmp_path, available=[d.isoformat() for d in days])
    src.sync()

    bars = src.load_bars()
    assert list(bars.columns) == sources.REQUIRED_COLUMNS
    assert len(bars) == len(days)
    assert src.head_key() == days[-1].isoformat()
    assert src.fingerprint()


def test_load_bars_without_cache_is_an_actionable_error(tmp_path):
    src = _fake(tmp_path, available=[])
    with pytest.raises(ValueError, match='data_update'):
        src.load_bars()


# --------------------------------------------------------------------------
# DCE auth
# --------------------------------------------------------------------------

def test_missing_credentials_name_both_variables(monkeypatch):
    monkeypatch.delenv('DCE_API_KEY', raising=False)
    monkeypatch.delenv('DCE_SECRET', raising=False)
    with pytest.raises(RuntimeError, match='DCE_API_KEY.*DCE_SECRET'):
        _DceAuth.credentials()


def test_missing_credentials_fail_the_sync_immediately(tmp_path, monkeypatch):
    """A config error must not degrade into per-day warnings and a quiet exit."""
    monkeypatch.delenv('DCE_API_KEY', raising=False)
    monkeypatch.delenv('DCE_SECRET', raising=False)
    src = make_source(_DceSource, 'C', tmp_path)
    src._auth = _DceAuth()

    def explode(day):
        raise AssertionError('sync must bail before requesting any day')

    monkeypatch.setattr(src, '_fetch_day', explode)
    with pytest.raises(RuntimeError, match='DCE_API_KEY'):
        src.sync()


@pytest.fixture
def dce_env(monkeypatch):
    monkeypatch.setenv('DCE_API_KEY', 'key')
    monkeypatch.setenv('DCE_SECRET', 'secret')


def test_auth_reuses_a_live_token(dce_env, monkeypatch):
    calls = []

    def fake_post(url, payload, headers=None):
        calls.append(url)
        return {'success': True, 'code': 200,
                'data': {'token': 'T', 'tokenType': 'Bearer', 'expiresIn': 28800}}

    monkeypatch.setattr(sources, '_post_json', fake_post)
    auth = _DceAuth()
    assert auth.token() == 'T'
    assert auth.token() == 'T'
    assert len(calls) == 1


def test_expired_token_triggers_one_relogin_and_retry(tmp_path, dce_env, monkeypatch):
    responses = [
        {'success': True, 'code': 200, 'data': {'token': 'T1', 'expiresIn': 28800}},
        {'success': False, 'code': 402, 'msg': 'token 过期'},
        {'success': True, 'code': 200, 'data': {'token': 'T2', 'expiresIn': 28800}},
        {'success': True, 'code': 200, 'data': [{'contractId': 'c2601'}]},
    ]
    seen = []

    def fake_post(url, payload, headers=None):
        seen.append(url)
        return responses.pop(0)

    monkeypatch.setattr(sources, '_post_json', fake_post)
    src = make_source(_DceSource, 'C', tmp_path)
    src._auth = _DceAuth()

    body = src._call('/forward/publicweb/dailystat/dayQuotes', {})
    assert body['data'] == [{'contractId': 'c2601'}]
    assert sum('accessToken' in u for u in seen) == 2      # initial + one re-login


def test_rate_limiting_backs_off_and_slows_the_throttle(tmp_path, dce_env, monkeypatch):
    slept = []
    monkeypatch.setattr(sources.time, 'sleep', slept.append)
    responses = [
        {'success': True, 'code': 200, 'data': {'token': 'T', 'expiresIn': 28800}},
        {'success': False, 'code': 501, 'msg': '访问过于频繁'},
        {'success': True, 'code': 200, 'data': [{'contractId': 'c2601'}]},
    ]
    monkeypatch.setattr(sources, '_post_json',
                        lambda url, payload, headers=None: responses.pop(0))

    src = make_source(_DceSource, 'C', tmp_path)
    src._auth = _DceAuth()
    before = src._throttle

    src._call('/forward/publicweb/dailystat/dayQuotes', {})
    assert slept == [5]
    assert src._throttle > before


def test_throttle_eases_back_after_a_clean_streak(tmp_path, dce_env, monkeypatch):
    """One burst of 501s must not slow every remaining day of a backfill."""
    monkeypatch.setattr(sources.time, 'sleep', lambda *_: None)
    ok = {'success': True, 'code': 200, 'data': [{'contractId': 'c2601'}]}
    monkeypatch.setattr(sources, '_post_json', lambda url, payload, headers=None:
                        {'success': True, 'code': 200,
                         'data': {'token': 'T', 'expiresIn': 28800}}
                        if 'accessToken' in url else ok)

    src = make_source(_DceSource, 'C', tmp_path)
    src._auth = _DceAuth()
    src._throttle = sources._DCE_MAX_THROTTLE
    for _ in range(sources._DCE_RECOVER_AFTER):
        src._call('/forward/publicweb/dailystat/dayQuotes', {})

    assert src._throttle < sources._DCE_MAX_THROTTLE
    assert src._throttle >= src.throttle          # never below the base rate


def test_a_failed_call_reports_the_exchange_message(tmp_path, dce_env, monkeypatch):
    responses = [
        {'success': True, 'code': 200, 'data': {'token': 'T', 'expiresIn': 28800}},
        {'success': False, 'code': 400, 'msg': '参数错误'},
    ]
    monkeypatch.setattr(sources, '_post_json',
                        lambda url, payload, headers=None: responses.pop(0))
    src = make_source(_DceSource, 'C', tmp_path)
    src._auth = _DceAuth()
    with pytest.raises(RuntimeError, match='参数错误'):
        src._call('/forward/publicweb/dailystat/dayQuotes', {})


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

@pytest.mark.parametrize('symbol, cls', [
    ('SA', sources._CzceSource),
    ('RB', _ShfeSource),
    ('C', _DceSource),
])
def test_get_source_dispatches_on_exchange(tmp_path, symbol, cls):
    assert isinstance(get_source(symbol, PRODUCTS[symbol], str(tmp_path)), cls)


def test_get_source_rejects_an_unknown_exchange(tmp_path):
    meta = dict(PRODUCTS['SA'], exchange='GFEX')
    with pytest.raises(NotImplementedError, match='GFEX'):
        get_source('SI', meta, str(tmp_path))


def test_shfe_and_dce_share_one_cache_dir_across_products(tmp_path):
    rb = make_source(_ShfeSource, 'RB', tmp_path)
    ag = make_source(_ShfeSource, 'AG', tmp_path)
    assert rb.cache_path(date(2026, 1, 5)) == ag.cache_path(date(2026, 1, 5))

    corn = make_source(_DceSource, 'C', tmp_path)
    coal = make_source(_DceSource, 'JM', tmp_path)
    assert corn.cache_path(date(2026, 1, 5)) == coal.cache_path(date(2026, 1, 5))
    assert corn.cache_path(date(2026, 1, 5)) != rb.cache_path(date(2026, 1, 5))


def test_every_exchange_caches_under_its_own_subfolder(tmp_path):
    """No venue writes into the top of ``cache/`` -- that holds only meta files."""
    soda = make_source(sources._CzceSource, 'SA', tmp_path)
    cotton = make_source(sources._CzceSource, 'CF', tmp_path)
    assert soda.cache_path(2026) == str(tmp_path / 'CZCE' / 'SA2026.txt')
    assert cotton.cache_path(2026) == str(tmp_path / 'CZCE' / 'CF2026.txt')

    rb = make_source(_ShfeSource, 'RB', tmp_path)
    corn = make_source(_DceSource, 'C', tmp_path)
    assert rb.cache_path(date(2026, 1, 5)) == str(
        tmp_path / 'SHFE' / 'kx20260105.json.gz')
    assert corn.cache_path(date(2026, 1, 5)) == str(
        tmp_path / 'DCE' / 'dayQuotes20260105.json.gz')


# --------------------------------------------------------------------------
# Download failures must be reported, never swallowed
# --------------------------------------------------------------------------

def test_a_failed_day_is_reported_as_a_failure(tmp_path):
    """A fetch error leaves the product behind the exchange, so it is named.

    Holidays are not failures: the exchange answering "no session" is a
    complete answer, and treating it as one would flag every run.
    """
    traded, holiday, flaky = _early_weekdays(3)
    src = _fake(tmp_path, available=[traded.isoformat()], broken=[flaky.isoformat()])
    src.sync()
    assert src.failures == [flaky.isoformat()]


def test_a_clean_sync_reports_no_failures(tmp_path):
    traded, holiday, _ = _early_weekdays(3)
    src = _fake(tmp_path, available=[traded.isoformat()])
    src.sync()
    assert src.failures == []


def test_failures_do_not_accumulate_across_syncs(tmp_path):
    """``failures`` describes the last sync, not every sync ever run."""
    traded, _, flaky = _early_weekdays(3)
    src = _fake(tmp_path, available=[traded.isoformat()], broken=[flaky.isoformat()])
    src.sync()
    assert src.failures == [flaky.isoformat()]

    src.broken = set()
    src.available = {traded.isoformat(), flaky.isoformat()}
    src.sync()
    assert src.failures == []


def test_czce_reports_a_failed_year_but_keeps_the_cache(tmp_path, monkeypatch):
    src = make_source(sources._CzceSource, 'SA', tmp_path)
    src.year = src.start_year                     # one year to try, and it fails
    path = src.cache_path(src.year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(b'x' * 128)                      # a usable cache already on disk

    def boom(url, dest, headers=None):
        raise urllib.error.URLError('down')

    monkeypatch.setattr(sources, '_http_download', boom)
    assert src.sync() == []
    assert src.failures == [str(src.year)]
    assert os.path.getsize(path) == 128           # stale bytes kept, not truncated


def test_czce_current_year_not_yet_published_is_not_a_failure(tmp_path, monkeypatch):
    """In early January the new year's file 404s before the first settlement.

    Nothing is cached for it, so nothing is stale -- the previous years are
    complete and the run should not report a problem.
    """
    src = make_source(sources._CzceSource, 'SA', tmp_path)
    src.year = src.start_year

    def missing(url, dest, headers=None):
        raise NoDataForDate(url)

    monkeypatch.setattr(sources, '_http_download', missing)
    assert src.sync() == []
    assert src.failures == []


def test_czce_missing_history_year_is_a_failure(tmp_path, monkeypatch):
    """A past year with nothing cached is a hole in the history, not a non-event."""
    src = make_source(sources._CzceSource, 'SA', tmp_path)
    src.year = src.start_year + 1                 # start_year is now a past year

    def missing(url, dest, headers=None):
        raise NoDataForDate(url)

    monkeypatch.setattr(sources, '_http_download', missing)
    src.sync()
    assert str(src.start_year) in src.failures
