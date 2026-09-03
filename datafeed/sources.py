"""Per-exchange market-data sources.

CZCE publishes one pipe-delimited text file per product per year. SHFE and DCE
both publish one JSON payload per *trading day* covering every product at once.
That difference in granularity is why this module exists: ``DataUpdate`` owns
cleaning, OI weighting, and CSV writing, and delegates "give me the raw bars for
this product" to an ``ExchangeSource``.

Every source returns the same thing from ``load_bars()``: a cleaned, de-duplicated,
date-sorted frame with exactly ``REQUIRED_COLUMNS``. Sources hand string values to
``_clean_contract_bars`` -- see ``_text`` for why that matters.

Cache layout::

    cache/CZCE/{SYM}{YEAR}.txt         CZCE, one file per product-year
    cache/SHFE/kx{YYYYMMDD}.json.gz    shared by every SHFE product
    cache/DCE/dayQuotes{YYYYMMDD}.json.gz   shared by every DCE product
    cache/{SHFE,DCE}/_calendar.json    self-built trading calendar

Every venue keeps its payloads in ``cache/{EXCHANGE}/``; only the per-product
``{SYM}.meta.json`` rebuild bookkeeping written by ``DataUpdate`` sits at the
top of ``cache/``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'date', 'contract', 'open', 'high', 'low', 'close', 'oi', 'volume', 'settle',
]
_NUMERIC_COLUMNS = [
    'prev_settle', 'open', 'high', 'low', 'close', 'settle',
    'change1', 'change2', 'volume', 'oi', 'oi_change', 'turnover', 'delivery_settle',
]

_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
)
_TIMEOUT = 45
_RETRIES = 3

_ISO = '%Y-%m-%d'
_COMPACT = '%Y%m%d'

# The exchanges can revise a day's numbers after first publishing them, so the
# most recent few days are always re-fetched instead of trusted from cache.
_REFETCH_TAIL_DAYS = 3


class NoDataForDate(Exception):
    """The exchange has no bars for this calendar day (weekend or holiday)."""


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _http_bytes(url: str, headers: dict = None, min_bytes: int = 0) -> bytes:
    """GET ``url`` and return the body.

    A 404 raises :class:`NoDataForDate` -- most SHFE holidays answer that way.
    400/403 fail fast; everything else is retried with backoff.

    ``min_bytes`` rejects a suspiciously short body as a failed transfer. Only
    the CZCE text files use it: a short JSON body is a perfectly valid "no
    session today" answer, and the daily sources decide that semantically.
    """
    request = urllib.request.Request(
        url, headers={'User-Agent': _USER_AGENT, **(headers or {})}
    )
    last_error = None
    for attempt in range(1, _RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as resp:
                payload = resp.read()
            if not payload or len(payload) < min_bytes:
                raise ValueError(f'Empty response from {url}')
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NoDataForDate(url) from exc
            last_error = exc
            if exc.code in (400, 403):
                break
            if attempt < _RETRIES:
                time.sleep(1.5 ** attempt)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < _RETRIES:
                time.sleep(1.5 ** attempt)
    raise last_error


_CZCE_MIN_BYTES = 64


def _http_download(url: str, dest: str, headers: dict = None) -> None:
    """Fetch ``url`` into ``dest`` atomically."""
    payload = _http_bytes(url, headers, min_bytes=_CZCE_MIN_BYTES)
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    tmp_path = f'{dest}.tmp'
    try:
        with open(tmp_path, 'wb') as fh:
            fh.write(payload)
        os.replace(tmp_path, dest)
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _post_json(url: str, payload: dict, headers: dict = None) -> dict:
    """POST a JSON body and decode the JSON response."""
    body = json.dumps(payload).encode('utf-8')
    hdrs = {
        'User-Agent': _USER_AGENT,
        'Content-Type': 'application/json',
        **(headers or {}),
    }
    last_error = None
    for attempt in range(1, _RETRIES + 1):
        try:
            request = urllib.request.Request(url, data=body, headers=hdrs, method='POST')
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < _RETRIES:
                time.sleep(1.5 ** attempt)
    raise last_error


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Shared cleaning
# --------------------------------------------------------------------------

def _text(value) -> str:
    """Render a raw JSON scalar as a string, the way CZCE text files arrive.

    ``_clean_contract_bars`` strips every object column with ``.str.strip()``,
    and that accessor turns non-string elements into NaN. SHFE's ``VOLUME`` and
    DCE's ``volumn`` are JSON integers, so leaving them unconverted would wipe
    them. Handing every source's values over as strings keeps one cleaning path
    with one dtype contract.
    """
    return '' if value is None else str(value)


def _clean_contract_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw string frame to typed ``REQUIRED_COLUMNS`` bars."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns: {missing}')

    out = df.copy()
    object_cols = out.select_dtypes(include=['object', 'string']).columns
    for col in object_cols:
        out[col] = out[col].str.strip()

    for col in _NUMERIC_COLUMNS:
        if col not in out.columns:
            continue
        series = (
            out[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
            .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, '-': pd.NA})
        )
        out[col] = pd.to_numeric(series, errors='coerce')

    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out['contract'] = out['contract'].astype(str).str.replace(' ', '', regex=False)
    out = out.dropna(subset=['date', 'contract'])
    out = out[~out['contract'].str.lower().isin(['', 'nan', 'none'])]
    out = out.drop_duplicates(subset=['date', 'contract'], keep='last')
    out = out.sort_values(['date', 'contract']).reset_index(drop=True)
    return out[REQUIRED_COLUMNS]


def _dedupe_sorted(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop_duplicates(subset=['date', 'contract'], keep='last')
    return data.sort_values(['date', 'contract']).reset_index(drop=True)


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------

class ExchangeSource:
    """Fetches and caches one product's raw bars for a single exchange."""

    exchange = ''

    def __init__(self, symbol: str, meta: dict, cache_dir: str) -> None:
        self.symbol = symbol
        self.meta = meta
        self.cache_dir = cache_dir
        self.dir = os.path.join(cache_dir, self.exchange)
        self.start_year = int(meta['start_year'])
        # Keys (years / dates) this exchange refused to hand over on the last
        # sync. A download failure is survivable -- the stale cache is still
        # usable and the next run retries -- but it must not pass for success:
        # the rebuild that follows would quietly write yesterday's numbers and
        # exit 0. ``sync`` clears this, and ``DataUpdate`` reports it.
        self.failures: List[str] = []

    def sync(self, force: bool = False) -> List[str]:
        """Refresh the local cache; return the keys (years / dates) refreshed.

        Also resets and repopulates ``self.failures``.
        """
        raise NotImplementedError

    def head_key(self) -> Optional[str]:
        """The newest cache unit's key -- the one ``fingerprint`` covers."""
        raise NotImplementedError

    def fingerprint(self) -> Optional[str]:
        """Digest of the newest cache unit, used to skip a needless rebuild."""
        raise NotImplementedError

    def load_bars(self) -> pd.DataFrame:
        """Every cached bar for this product, cleaned and sorted."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# CZCE -- one text file per product-year
# --------------------------------------------------------------------------

_CZCE_HEADERS = {'Referer': 'https://www.czce.com.cn/'}

_CZCE_HEADER_MAP = {
    '交易日期': 'date',
    '合约代码': 'contract',
    '品种代码': 'contract',
    '昨结算': 'prev_settle',
    '今开盘': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '今收盘': 'close',
    '今结算': 'settle',
    '涨跌1': 'change1',
    '涨跌2': 'change2',
    '成交量(手)': 'volume',
    '持仓量': 'oi',
    '空盘量': 'oi',
    '增减量': 'oi_change',
    '成交额(万元)': 'turnover',
    '交割结算价': 'delivery_settle',
}


def _czce_url(symbol: str, year: int) -> str:
    base = f'https://www.czce.com.cn/cn/DFSStaticFiles/Future/{year}/FutureDataAllHistory'
    if year < 2020:
        return f'{base}/{symbol}.txt'
    return f'{base}/{symbol}FUTURES{year}.txt'


def _detect_encoding(raw: bytes) -> str:
    for encoding in ('utf-8-sig', 'gb18030', 'gbk'):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return 'utf-8'


def _normalize_columns(columns) -> list:
    mapped = []
    seen = {}
    for col in columns:
        key = str(col).replace('﻿', '').strip().replace(' ', '')
        name = _CZCE_HEADER_MAP.get(key, key)
        if name in seen:
            name = f'{name}_{seen[name]}'
        seen[name] = seen.get(name, 0) + 1
        mapped.append(name)
    return mapped


def _read_czce_history(path: str) -> pd.DataFrame:
    with open(path, 'rb') as fh:
        raw = fh.read()
    if not raw.strip():
        raise ValueError(f'Empty file: {path}')

    text = raw.decode(_detect_encoding(raw), errors='replace')
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines[:12]) if '交易日期' in line and '|' in line),
        None,
    )
    if header_idx is None:
        raise ValueError(f'No CZCE header found in {path}')

    body = []
    for line in lines[header_idx:]:
        stripped = line.strip().rstrip('|').strip()
        if stripped:
            body.append(stripped)
    if len(body) < 2:
        raise ValueError(f'No rows parsed from {path}')

    headers = [h.strip() for h in body[0].split('|')]
    while headers and headers[-1] == '':
        headers.pop()
    n_cols = len(headers)
    rows = []
    for line in body[1:]:
        parts = [p.strip() for p in line.split('|')]
        while parts and parts[-1] == '':
            parts.pop()
        if len(parts) < 2:
            continue
        if len(parts) < n_cols:
            parts.extend([''] * (n_cols - len(parts)))
        rows.append(parts[:n_cols])

    df = pd.DataFrame(rows, columns=headers, dtype=str)
    df.columns = _normalize_columns(df.columns)
    df = df.loc[:, ~df.columns.str.match(r'^(Unnamed|$)')]
    df = df.dropna(how='all')
    if df.empty:
        raise ValueError(f'No rows parsed from {path}')
    return df


class _CzceSource(ExchangeSource):
    exchange = 'CZCE'

    def __init__(self, symbol: str, meta: dict, cache_dir: str) -> None:
        super().__init__(symbol, meta, cache_dir)
        self.year = datetime.now().year

    def years(self) -> range:
        return range(self.start_year, self.year + 1)

    def cache_path(self, year: int) -> str:
        return os.path.join(self.dir, f'{self.symbol}{year}.txt')

    def head_key(self) -> str:
        return str(self.year)

    def sync(self, force: bool = False) -> List[str]:
        os.makedirs(self.dir, exist_ok=True)
        self.failures = []
        refreshed = []
        for year in self.years():
            path = self.cache_path(year)
            is_current = year == self.year
            if (
                not force
                and not is_current
                and os.path.exists(path)
                and os.path.getsize(path) > 64
            ):
                logger.info('%s%d cache hit, skip download.', self.symbol, year)
                continue
            url = _czce_url(self.symbol, year)
            try:
                _http_download(url, path, _CZCE_HEADERS)
                refreshed.append(str(year))
                logger.info('%s%d Update Done.', self.symbol, year)
            except (
                NoDataForDate,
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                OSError,
            ) as exc:
                if os.path.exists(path) and os.path.getsize(path) > 64:
                    self.failures.append(str(year))
                    logger.warning(
                        '%s%d Update Error (%s); keep existing cache.',
                        self.symbol, year, exc,
                    )
                elif is_current and isinstance(exc, NoDataForDate):
                    # Between New Year's Day and the year's first settlement
                    # CZCE has not posted the file yet. Nothing is cached, so
                    # nothing can be stale -- last year's bars are complete.
                    logger.info('%s%d not published yet.', self.symbol, year)
                else:
                    self.failures.append(str(year))
                    logger.warning('%s%d Update Error (%s).', self.symbol, year, exc)
        return refreshed

    def fingerprint(self) -> Optional[str]:
        path = self.cache_path(self.year)
        return _file_sha256(path) if os.path.exists(path) else None

    def load_bars(self) -> pd.DataFrame:
        frames = []
        for year in self.years():
            path = self.cache_path(year)
            if not os.path.exists(path):
                logger.warning('%s does not exist, skipping ...', path)
                continue
            try:
                frames.append(_clean_contract_bars(_read_czce_history(path)))
            except (ValueError, KeyError) as exc:
                logger.warning('%s skipped (%s)', path, exc)
                continue
        if not frames:
            raise ValueError(
                f'No usable cache files for {self.symbol}; '
                f'need columns {REQUIRED_COLUMNS}'
            )
        return _dedupe_sorted(pd.concat(frames, ignore_index=True))


# --------------------------------------------------------------------------
# Shared base for the per-trading-day exchanges (SHFE, DCE)
# --------------------------------------------------------------------------

class _DailyFileSource(ExchangeSource):
    """One cached payload per trading day, shared by every product on the venue.

    The payloads are keyed by date and hold every product, so syncing RB then AG
    then C downloads each day exactly once -- the 2nd and 3rd products are pure
    cache hits. Neither exchange publishes a holiday calendar, so we build one:
    weekends are skipped without a request, and any other date that comes back
    empty is recorded in ``_calendar.json`` and never requested again.
    """

    prefix = ''
    throttle = 0.25

    def __init__(self, symbol: str, meta: dict, cache_dir: str) -> None:
        super().__init__(symbol, meta, cache_dir)
        self._calendar_path = os.path.join(self.dir, '_calendar.json')
        self._throttle = self.throttle

    # -- subclass hooks ----------------------------------------------------

    def _fetch_day(self, day: date) -> bytes:
        """Raw payload for one day, or raise :class:`NoDataForDate`."""
        raise NotImplementedError

    def _extract(self, payload: dict, day: date) -> List[dict]:
        """This product's rows for one day, as ``REQUIRED_COLUMNS`` dicts."""
        raise NotImplementedError

    # -- cache -------------------------------------------------------------

    def cache_path(self, day: date) -> str:
        return os.path.join(self.dir, f'{self.prefix}{day:%Y%m%d}.json.gz')

    def _write_day(self, path: str, payload: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f'{path}.tmp'
        try:
            with gzip.open(tmp_path, 'wb') as fh:
                fh.write(payload)
            os.replace(tmp_path, path)
        except OSError:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def _read_day(path: str) -> dict:
        with gzip.open(path, 'rb') as fh:
            return json.loads(fh.read().decode('utf-8'))

    def _cached_payload(self, day: date) -> Optional[bytes]:
        """The bytes already cached for ``day``, or None."""
        path = self.cache_path(day)
        if not os.path.exists(path):
            return None
        try:
            with gzip.open(path, 'rb') as fh:
                return fh.read()
        except OSError:
            return None

    def _cached_days(self) -> List[tuple]:
        if not os.path.isdir(self.dir):
            return []
        pattern = re.compile(rf'^{re.escape(self.prefix)}(\d{{8}})\.json\.gz$')
        out = []
        for name in os.listdir(self.dir):
            match = pattern.match(name)
            if not match:
                continue
            try:
                day = datetime.strptime(match.group(1), _COMPACT).date()
            except ValueError:
                continue
            if day.year < self.start_year:
                continue
            out.append((day, os.path.join(self.dir, name)))
        out.sort()
        return out

    def _read_calendar(self) -> dict:
        if not os.path.exists(self._calendar_path):
            return {}
        try:
            with open(self._calendar_path, encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_calendar(self, state: dict) -> None:
        os.makedirs(self.dir, exist_ok=True)
        tmp_path = f'{self._calendar_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._calendar_path)

    # -- ExchangeSource ----------------------------------------------------

    def sync(self, force: bool = False) -> List[str]:
        os.makedirs(self.dir, exist_ok=True)
        self.failures = []
        state = self._read_calendar()
        no_data = set(state.get('no_data') or ())

        today = date.today()
        tail_start = today - timedelta(days=_REFETCH_TAIL_DAYS)
        day = date(self.start_year, 1, 1)
        refreshed: List[str] = []
        downloaded = 0

        while day <= today:
            if day.weekday() >= 5:                 # no weekend session to ask about
                day += timedelta(days=1)
                continue
            key = day.isoformat()
            settled = day < tail_start
            if not force and settled:
                if key in no_data or os.path.exists(self.cache_path(day)):
                    day += timedelta(days=1)
                    continue
            try:
                payload = self._fetch_day(day)
            except NoDataForDate:
                no_data.add(key)
                time.sleep(self._throttle)
                day += timedelta(days=1)
                continue
            except Exception as exc:               # noqa: BLE001 -- retry next run
                self.failures.append(key)
                logger.warning(
                    '%s %s fetch failed (%s); left for the next run.',
                    self.exchange, key, exc,
                )
                time.sleep(self._throttle)
                day += timedelta(days=1)
                continue
            no_data.discard(key)
            downloaded += 1
            # The trailing days are re-fetched every run in case they were
            # revised. Only a day whose bytes actually moved counts as
            # refreshed, so an unchanged re-fetch does not force a rebuild.
            if payload != self._cached_payload(day):
                self._write_day(self.cache_path(day), payload)
                refreshed.append(key)
            if downloaded % 100 == 0:
                logger.info('%s %s downloaded (%d days so far)', self.exchange, key, downloaded)
            time.sleep(self._throttle)
            day += timedelta(days=1)

        state['no_data'] = sorted(no_data)
        state['last_checked'] = today.isoformat()
        self._write_calendar(state)
        if downloaded:
            logger.info('%s %s: %d day(s) downloaded.', self.exchange, self.symbol, downloaded)
        return refreshed

    def head_key(self) -> Optional[str]:
        days = self._cached_days()
        return days[-1][0].isoformat() if days else None

    def fingerprint(self) -> Optional[str]:
        days = self._cached_days()
        if not days:
            return None
        day, path = days[-1]
        digest = hashlib.sha256()
        digest.update(f'{day.isoformat()}:{_file_sha256(path)}'.encode('utf-8'))
        return digest.hexdigest()

    def load_bars(self) -> pd.DataFrame:
        rows: List[dict] = []
        for day, path in self._cached_days():
            try:
                payload = self._read_day(path)
            except (OSError, ValueError) as exc:
                logger.warning('%s skipped (%s)', path, exc)
                continue
            rows.extend(self._extract(payload, day))
        if not rows:
            raise ValueError(
                f'No usable {self.exchange} cache for {self.symbol}; '
                f'run `python -m datafeed.data_update {self.symbol}` to download.'
            )
        frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
        return _dedupe_sorted(_clean_contract_bars(frame))


# --------------------------------------------------------------------------
# SHFE -- unauthenticated daily JSON
# --------------------------------------------------------------------------

_SHFE_URL = 'https://www.shfe.com.cn/data/tradedata/future/dailydata/kx{day:%Y%m%d}.dat'
_SHFE_HEADERS = {'Referer': 'https://www.shfe.com.cn/'}
_SHFE_FIELDS = {
    'open': 'OPENPRICE',
    'high': 'HIGHESTPRICE',
    'low': 'LOWESTPRICE',
    'close': 'CLOSEPRICE',
    'settle': 'SETTLEMENTPRICE',
    'oi': 'OPENINTEREST',
    'volume': 'VOLUME',
}


class _ShfeSource(_DailyFileSource):
    """Shanghai Futures Exchange daily bars.

    A non-trading day is announced two different ways: most answer 404, but some
    (New Year's Day, National Day) answer 200 with a 31-byte
    ``{"o_curinstrument": []}``. Both mean the same thing, so emptiness is
    decided on the parsed payload rather than on the response size -- otherwise
    those days look like transient failures and get re-requested forever.

    Rows carry a ``PRODUCTID`` like ``rb_f``; an exact match on that also
    excludes the non-futures rows (``auefp``, ``sc_tas``, ``op_f``), and dropping
    non-numeric ``DELIVERYMONTH`` values drops the per-product ``小计`` and
    whole-table ``总计`` subtotals.
    """

    exchange = 'SHFE'
    prefix = 'kx'

    def _fetch_day(self, day: date) -> bytes:
        payload = _http_bytes(_SHFE_URL.format(day=day), _SHFE_HEADERS)
        body = json.loads(payload.decode('utf-8'))
        if not body.get('o_curinstrument'):
            raise NoDataForDate(day.isoformat())
        return payload

    def _extract(self, payload: dict, day: date) -> List[dict]:
        product_id = f'{self.symbol.lower()}_f'
        stamp = day.strftime(_ISO)
        rows = []
        for raw in payload.get('o_curinstrument') or ():
            if str(raw.get('PRODUCTID') or '').strip() != product_id:
                continue
            month = str(raw.get('DELIVERYMONTH') or '').strip()
            if not month.isdigit():
                continue
            row = {'date': stamp, 'contract': f'{self.symbol}{month}'}
            for col, key in _SHFE_FIELDS.items():
                row[col] = _text(raw.get(key))
            rows.append(row)
        return rows


# --------------------------------------------------------------------------
# DCE -- authenticated open API
# --------------------------------------------------------------------------

_DCE_BASE = 'http://www.dce.com.cn/dceapi'
_DCE_ENV_KEY = 'DCE_API_KEY'
_DCE_ENV_SECRET = 'DCE_SECRET'
_DCE_TOKEN_EXPIRED = 402
_DCE_RATE_LIMITED = 501
_DCE_BACKOFF = (5, 15, 45)
_DCE_MAX_THROTTLE = 3.0
# Rate limiting is bursty, so the throttle has to come back down as well as up:
# without recovery one burst of 501s would slow every remaining day of a backfill.
_DCE_RECOVER_AFTER = 50
_DCE_RECOVER_FACTOR = 0.8

# Any listed futures contract, used only to tell a trading day from a holiday.
_DCE_ANY_CONTRACT = re.compile(r'^[a-z]+\d{4}$')

_DCE_FIELDS = {
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'settle': 'clearPrice',
    'oi': 'openInterest',
    'volume': 'volumn',      # the exchange really does spell it this way
}


class _DceAuth:
    """apikey + Bearer token for the DCE open API.

    Credentials come from the environment; the token lives in memory only and is
    never written to the cache or the log. One instance is shared by every DCE
    product in a run, so a single login covers the whole sync.
    """

    def __init__(self) -> None:
        self._token = None
        self._expires_at = 0.0

    @staticmethod
    def credentials() -> tuple:
        key = os.environ.get(_DCE_ENV_KEY)
        secret = os.environ.get(_DCE_ENV_SECRET)
        if not key or not secret:
            raise RuntimeError(
                f'DCE products need {_DCE_ENV_KEY} and {_DCE_ENV_SECRET} in the '
                f'environment. CZCE and SHFE products do not need them.'
            )
        return key, secret

    def _login(self) -> str:
        key, secret = self.credentials()
        payload = _post_json(
            f'{_DCE_BASE}/cms/auth/accessToken', {'secret': secret}, {'apikey': key}
        )
        data = payload.get('data') or {}
        token = data.get('token')
        if not token:
            raise RuntimeError(
                f"DCE login failed: code={payload.get('code')} msg={payload.get('msg')!r}"
            )
        # The published docs say 3600s; the live API returns 28800. Trust the
        # response and renew a minute early either way.
        ttl = float(data.get('expiresIn') or 3600)
        self._token = token
        self._expires_at = time.time() + max(ttl - 60, 60)
        logger.info('DCE token issued (valid %.0fs).', ttl)
        return token

    def token(self) -> str:
        if self._token and time.time() < self._expires_at:
            return self._token
        return self._login()

    def headers(self) -> dict:
        key, _ = self.credentials()
        return {'apikey': key, 'Authorization': f'Bearer {self.token()}'}

    def invalidate(self) -> None:
        self._token = None
        self._expires_at = 0.0


class _DceSource(_DailyFileSource):
    """Dalian Commodity Exchange daily bars, via the authenticated open API.

    DCE's unauthenticated ``/publicweb/`` and ``/dcereport/`` paths sit behind a
    JS anti-bot challenge and always answer 412; the ``/dceapi/`` gateway used
    here does not.

    A non-trading day still answers 200 -- with a single ``总计`` summary row
    whose ``contractId`` is null -- so an empty ``data`` array is not the test
    for a holiday; the absence of any real contract is.

    Product filtering must be an anchored regex, never a prefix match: on any
    given day the payload carries ``c2609`` next to ``cs2609`` (corn starch) and
    ``jm2609`` next to ``j2609`` (coke).
    """

    exchange = 'DCE'
    prefix = 'dayQuotes'
    throttle = 0.3

    _auth = _DceAuth()

    def __init__(self, symbol: str, meta: dict, cache_dir: str) -> None:
        super().__init__(symbol, meta, cache_dir)
        self.variety = symbol.lower()
        self._contract_re = re.compile(rf'^{re.escape(self.variety)}\d{{4}}$')
        self._clean_streak = 0

    def sync(self, force: bool = False) -> List[str]:
        # Missing credentials are a configuration error, not a per-day transient:
        # check once up front so the run fails fast and loudly instead of warning
        # on every date and then quietly succeeding off a stale cache.
        self._auth.credentials()
        return super().sync(force=force)

    def _ease_throttle(self) -> None:
        """Walk the throttle back toward the base after a clean streak."""
        self._clean_streak += 1
        if self._clean_streak < _DCE_RECOVER_AFTER or self._throttle <= self.throttle:
            return
        self._clean_streak = 0
        self._throttle = max(self._throttle * _DCE_RECOVER_FACTOR, self.throttle)
        logger.info('DCE steady again; throttle eased to %.2fs.', self._throttle)

    def _call(self, path: str, payload: dict) -> dict:
        url = f'{_DCE_BASE}{path}'
        relogged = False
        for pause in (*_DCE_BACKOFF, None):
            body = _post_json(url, payload, self._auth.headers())
            code = body.get('code')
            if code == _DCE_TOKEN_EXPIRED and not relogged:
                relogged = True
                self._auth.invalidate()
                body = _post_json(url, payload, self._auth.headers())
                code = body.get('code')
            if code == _DCE_RATE_LIMITED:
                if pause is None:
                    break
                self._clean_streak = 0
                self._throttle = min(self._throttle * 2, _DCE_MAX_THROTTLE)
                logger.warning(
                    'DCE rate limited; sleeping %ss (throttle now %.2fs).',
                    pause, self._throttle,
                )
                time.sleep(pause)
                continue
            if not body.get('success'):
                raise RuntimeError(
                    f"DCE {path} failed: code={code} msg={body.get('msg')!r}"
                )
            self._ease_throttle()
            return body
        raise RuntimeError(f'DCE {path} still rate limited after {len(_DCE_BACKOFF)} retries')

    def _fetch_day(self, day: date) -> bytes:
        body = self._call(
            '/forward/publicweb/dailystat/dayQuotes',
            {
                'varietyId': 'all',
                'tradeDate': day.strftime(_COMPACT),
                'tradeType': '1',
                'lang': 'zh',
            },
        )
        rows = body.get('data') or []
        if not any(
            _DCE_ANY_CONTRACT.match(str(row.get('contractId') or '').strip())
            for row in rows
        ):
            raise NoDataForDate(day.isoformat())
        return json.dumps(body, ensure_ascii=False).encode('utf-8')

    def _extract(self, payload: dict, day: date) -> List[dict]:
        stamp = day.strftime(_ISO)
        rows = []
        for raw in payload.get('data') or ():
            code = str(raw.get('contractId') or '').strip()
            if not self._contract_re.match(code):
                continue
            row = {'date': stamp, 'contract': code.upper()}
            for col, key in _DCE_FIELDS.items():
                row[col] = _text(raw.get(key))
            rows.append(row)
        return rows


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

_SOURCES: Dict[str, type] = {
    'CZCE': _CzceSource,
    'SHFE': _ShfeSource,
    'DCE': _DceSource,
}


def get_source(symbol: str, meta: dict, cache_dir: str) -> ExchangeSource:
    """Build the data source for a product, dispatching on ``meta['exchange']``."""
    exchange = str(meta.get('exchange') or '').upper()
    source_cls = _SOURCES.get(exchange)
    if source_cls is None:
        supported = ', '.join(sorted(_SOURCES))
        raise NotImplementedError(
            f'Exchange {exchange!r} is not supported; known: {supported}'
        )
    return source_cls(symbol, meta, cache_dir)
