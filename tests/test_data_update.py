"""OI weighting and freshness reporting for datafeed.data_update.

Runs entirely on inline frames and a stub source -- no exchange is contacted.
"""

import json

import pandas as pd
import pytest

from datafeed import data_update
from datafeed.data_update import DataUpdate, _build_weighted


def _bars(rows) -> pd.DataFrame:
    """Contract-level bars from ``(date, contract, o, h, l, c, settle, oi, vol)``."""
    return pd.DataFrame(
        rows,
        columns=['date', 'contract', 'open', 'high', 'low', 'close',
                 'settle', 'oi', 'volume'],
    ).astype({'date': 'datetime64[ns]'})


# --------------------------------------------------------------------------
# OI weighting
# --------------------------------------------------------------------------

def test_weighting_is_open_interest_proportional():
    weighted = _build_weighted(_bars([
        ('2026-01-05', 'RB2601', 100, 110, 90, 105, 104, 300, 10),
        ('2026-01-05', 'RB2605', 200, 210, 190, 205, 204, 100, 10),
    ]))
    assert len(weighted) == 1
    assert weighted.loc[0, 'close'] == pytest.approx((105 * 300 + 205 * 100) / 400)
    assert weighted.loc[0, 'oi'] == 400
    assert weighted.loc[0, 'volume'] == 20


def test_a_zero_priced_contract_is_excluded():
    """Expiring contracts print ``open = high = low = 0`` while still trading.

    Weighting those in dragged the day's open/high/low toward zero while the
    close stayed honest, which is how a bar ended up with ``close > high``.
    """
    weighted = _build_weighted(_bars([
        ('2026-01-05', 'C2605', 2300, 2350, 2290, 2340, 2330, 100_000, 50_000),
        ('2026-01-05', 'C2601', 0, 0, 0, 2311, 2311, 3_000, 500),
    ]))
    row = weighted.loc[0]
    # the surviving contract's own bar, undiluted
    assert row['open'] == pytest.approx(2300)
    assert row['high'] == pytest.approx(2350)
    assert row['low'] == pytest.approx(2290)
    assert row['close'] == pytest.approx(2340)
    assert row['oi'] == 100_000          # the dropped row's OI leaves with it


@pytest.mark.parametrize('bad_price', [
    (0, 110, 90, 105, 104),      # no opening print
    (100, 0, 90, 105, 104),
    (100, 110, 0, 105, 104),
    (100, 110, 90, 0, 104),
    (100, 110, 90, 105, 0),      # settlement missing
    (-1, 110, 90, 105, 104),     # nonsense, whatever its source
])
def test_any_non_positive_price_disqualifies_the_row(bad_price):
    o, h, l, c, s = bad_price
    weighted = _build_weighted(_bars([
        ('2026-01-05', 'RB2601', 100, 110, 90, 105, 104, 300, 10),
        ('2026-01-05', 'RB2605', o, h, l, c, s, 900, 90),
    ]))
    assert weighted.loc[0, 'oi'] == 300


def test_the_weighted_bar_stays_internally_consistent():
    weighted = _build_weighted(_bars([
        ('2026-01-05', 'AG2602', 6000, 6100, 5900, 6050, 6040, 50_000, 20_000),
        ('2026-01-05', 'AG2604', 0, 0, 0, 6390, 6390, 2_000, 300),
        ('2026-01-06', 'AG2602', 6050, 6200, 6010, 6180, 6170, 51_000, 21_000),
    ]))
    assert len(weighted) == 2
    for _, row in weighted.iterrows():
        assert row['low'] <= row['open'] <= row['high']
        assert row['low'] <= row['close'] <= row['high']


def test_untraded_and_unlisted_rows_are_still_excluded():
    """The pre-existing liquidity filter is unchanged by the price filter."""
    weighted = _build_weighted(_bars([
        ('2026-01-05', 'RB2601', 100, 110, 90, 105, 104, 300, 10),
        ('2026-01-05', 'RB2605', 200, 210, 190, 205, 204, 0, 10),     # not listed
        ('2026-01-05', 'RB2610', 300, 310, 290, 305, 304, 500, 0),    # no trade
    ]))
    assert weighted.loc[0, 'oi'] == 300


def test_a_missing_price_is_excluded():
    weighted = _build_weighted(_bars([
        ('2026-01-05', 'RB2601', 100, 110, 90, 105, 104, 300, 10),
        ('2026-01-05', 'RB2605', 200, None, 190, 205, 204, 900, 90),
    ]))
    assert weighted.loc[0, 'oi'] == 300


def test_a_day_with_nothing_usable_is_an_error():
    with pytest.raises(ValueError, match='No rows left'):
        _build_weighted(_bars([
            ('2026-01-05', 'RB2601', 0, 0, 0, 105, 104, 300, 10),
        ]))


# --------------------------------------------------------------------------
# Freshness reporting
# --------------------------------------------------------------------------

class _StubSource:
    """Stands in for an ExchangeSource: canned bars and a canned failure list."""

    def __init__(self, failures=(), fingerprint='fp-1'):
        self.failures = list(failures)
        self._fingerprint = fingerprint
        self.synced = 0
        self.built = 0

    def sync(self, force=False):
        self.synced += 1
        return []

    def head_key(self):
        return '2026'

    def fingerprint(self):
        return self._fingerprint

    def load_bars(self):
        self.built += 1
        return _bars([
            ('2026-01-05', 'RB2601', 100, 110, 90, 105, 104, 300, 10),
        ])


def _meta(job) -> dict:
    with open(job._meta_path, encoding='utf-8') as fh:
        return json.load(fh)


def _job(tmp_path, source):
    job = DataUpdate('RB', data_dir=str(tmp_path / 'data'),
                     cache_dir=str(tmp_path / 'cache'))
    job.source = source
    return job


def test_a_partial_download_is_recorded_not_swallowed(tmp_path):
    """The CSVs still get built -- the point is that the caller is told."""
    job = _job(tmp_path, _StubSource(failures=['2026']))
    job.update()
    assert job.stale_keys == ['2026']
    assert pd.read_csv(job.weighted_path).shape[0] == 1     # still usable


def test_a_clean_download_reports_nothing_stale(tmp_path):
    job = _job(tmp_path, _StubSource())
    job.update()
    assert job.stale_keys == []


def test_rebuild_only_never_reports_staleness(tmp_path):
    """No sync ran, so a source's leftover failure list says nothing."""
    source = _StubSource(failures=['2026'])
    job = _job(tmp_path, source)
    job.update(rebuild_only=True)
    assert source.synced == 0
    assert job.stale_keys == []


def test_the_meta_file_marks_an_incomplete_build(tmp_path):
    job = _job(tmp_path, _StubSource(failures=['2026-01-05', '2026-01-06']))
    job.update()
    assert _meta(job)['stale_keys'] == ['2026-01-05', '2026-01-06']


def test_a_skipped_rebuild_still_records_the_current_run(tmp_path):
    """An unchanged fingerprint says the CSVs need no rebuild -- nothing more.

    The download can still have failed on this run (or have recovered from a
    previous one), and the meta file is where that is read back from.
    """
    source = _StubSource()
    job = _job(tmp_path, source)
    job.update()                                # builds, and records a clean run
    assert _meta(job)['stale_keys'] == []

    source.failures = ['2026']
    job.update()
    assert source.built == 1                    # the rebuild really was skipped
    assert _meta(job)['stale_keys'] == ['2026']

    source.failures = []                        # ... and it clears again
    job.update()
    assert source.built == 1
    assert _meta(job)['stale_keys'] == []


def test_cli_exits_non_zero_on_an_incomplete_download(tmp_path, monkeypatch):
    """A scheduled refresh must not report success on data it failed to fetch."""
    monkeypatch.setattr(data_update, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(data_update, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(
        data_update, 'get_source',
        lambda symbol, meta, cache_dir: _StubSource(failures=['2026']),
    )
    with pytest.raises(SystemExit) as excinfo:
        data_update.main(['RB'])
    message = str(excinfo.value)
    assert 'incomplete' in message
    assert 'RB' in message


def test_cli_stays_quiet_when_every_product_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(data_update, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(data_update, 'CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(
        data_update, 'get_source',
        lambda symbol, meta, cache_dir: _StubSource(),
    )
    data_update.main(['RB'])        # no SystemExit


def test_cli_separates_hard_failures_from_stale_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(data_update, 'DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setattr(data_update, 'CACHE_DIR', str(tmp_path / 'cache'))

    def source_for(symbol, meta, cache_dir):
        if symbol == 'RB':
            raise RuntimeError('registry blew up')
        return _StubSource(failures=['2026'])

    monkeypatch.setattr(data_update, 'get_source', source_for)
    with pytest.raises(SystemExit) as excinfo:
        data_update.main(['RB', 'AG'])
    message = str(excinfo.value)
    assert 'failed for: RB' in message
    assert 'incomplete' in message and 'AG' in message
