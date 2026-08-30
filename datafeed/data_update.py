"""
CZCE futures history download and OI-weighted aggregation.

Usage (from the repo root):
  python -m datafeed.data_update              # incremental: all registered products
  python -m datafeed.data_update FG CF        # selected products only
  python -m datafeed.data_update --force      # re-download every year
  python -m datafeed.data_update --rebuild-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd

from .products import get_product, list_products, normalize_symbol, require_products

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

_HEADER_MAP = {
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

_REQUIRED_COLUMNS = [
    'date', 'contract', 'open', 'high', 'low', 'close', 'oi', 'volume', 'settle',
]
_NUMERIC_COLUMNS = [
    'prev_settle', 'open', 'high', 'low', 'close', 'settle',
    'change1', 'change2', 'volume', 'oi', 'oi_change', 'turnover', 'delivery_settle',
]
_WEIGHTED_PRICE_COLS = ['open', 'high', 'low', 'close', 'settle']

_HTTP_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://www.czce.com.cn/',
}
_TIMEOUT = 45
_RETRIES = 3


def _czce_url(symbol: str, year: int) -> str:
    base = f'https://www.czce.com.cn/cn/DFSStaticFiles/Future/{year}/FutureDataAllHistory'
    if year < 2020:
        return f'{base}/{symbol}.txt'
    return f'{base}/{symbol}FUTURES{year}.txt'


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
        name = _HEADER_MAP.get(key, key)
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


def _clean_contract_bars(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
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
    return out[_REQUIRED_COLUMNS]


def _build_weighted(df: pd.DataFrame) -> pd.DataFrame:
    need = _WEIGHTED_PRICE_COLS + ['oi', 'volume']
    work = df.dropna(subset=need)
    work = work[(work['oi'] > 0) & (work['volume'] > 0)].copy()
    if work.empty:
        raise ValueError('No rows left to build the OI-weighted series')

    oi = work['oi']
    for col in _WEIGHTED_PRICE_COLS:
        work[f'_{col}'] = work[col] * oi

    grouped = work.groupby('date', sort=True)
    oi_sum = grouped['oi'].sum()
    result = pd.DataFrame({'date': oi_sum.index})
    for col in _WEIGHTED_PRICE_COLS:
        result[col] = (grouped[f'_{col}'].sum() / oi_sum).to_numpy()
    result['oi'] = oi_sum.to_numpy()
    result['volume'] = grouped['volume'].sum().to_numpy()
    return result.reset_index(drop=True)


def _to_csv_atomic(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = f'{path}.tmp'
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _http_download(url: str, dest: str) -> None:
    request = urllib.request.Request(url, headers=_HTTP_HEADERS)
    tmp_path = f'{dest}.tmp'
    last_error = None
    for attempt in range(1, _RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as resp:
                payload = resp.read()
            if not payload or len(payload) < 64:
                raise ValueError(f'Empty response from {url}')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(tmp_path, 'wb') as fh:
                fh.write(payload)
            os.replace(tmp_path, dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 403, 404):
                break
            if attempt < _RETRIES:
                time.sleep(1.5 ** attempt)
    raise last_error


class DataUpdate:
    def __init__(self, category: str, data_dir: str = None, cache_dir: str = None):
        meta = get_product(category)
        self.category = normalize_symbol(category)
        self.exchange = meta['exchange']
        self.start_year = meta['start_year']
        self.year = datetime.now().year
        self.data_dir = data_dir or DATA_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self.raw_path = os.path.join(self.data_dir, f'{self.category}.csv')
        self.weighted_path = os.path.join(self.data_dir, f'{self.category}_weighted.csv')
        self._meta_path = os.path.join(self.cache_dir, f'{self.category}.meta.json')

    def years(self) -> range:
        return range(self.start_year, self.year + 1)

    def cache_path(self, year: int) -> str:
        return os.path.join(self.cache_dir, f'{self.category}{year}.txt')

    def update(self, force: bool = False, rebuild_only: bool = False) -> pd.DataFrame:
        """Download CZCE history (incrementally) and rebuild contract / weighted CSVs."""
        if self.exchange != 'CZCE':
            raise NotImplementedError(f'Exchange {self.exchange} is not supported')

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        refreshed = []
        if not rebuild_only:
            refreshed = self._download_years(force=force)

        current_cache = self.cache_path(self.year)
        current_hash = _file_sha256(current_cache) if os.path.exists(current_cache) else None
        if self._is_up_to_date(force, rebuild_only, refreshed, current_hash):
            logger.info('%s already up to date; skip rebuild.', self.category)
            return pd.read_csv(self.raw_path, parse_dates=['date'])

        data = self._load_contract_bars()
        _to_csv_atomic(self._format_dates(data), self.raw_path)
        weighted = _build_weighted(data)
        _to_csv_atomic(self._format_dates(weighted), self.weighted_path)
        self._write_meta(current_hash, refreshed)
        logger.info(
            '%s saved %d contract rows / %d weighted days -> %s',
            self.category, len(data), len(weighted), self.data_dir,
        )
        return data

    def _download_years(self, force: bool) -> list:
        refreshed = []
        for year in self.years():
            path = self.cache_path(year)
            is_current = year == self.year
            if not force and not is_current and os.path.exists(path) and os.path.getsize(path) > 64:
                logger.info('%s%d cache hit, skip download.', self.category, year)
                continue
            url = _czce_url(self.category, year)
            try:
                _http_download(url, path)
                refreshed.append(year)
                logger.info('%s%d Update Done.', self.category, year)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                if os.path.exists(path) and os.path.getsize(path) > 64:
                    logger.warning('%s%d Update Error (%s); keep existing cache.', self.category, year, exc)
                else:
                    logger.warning('%s%d Update Error (%s).', self.category, year, exc)
        return refreshed

    def _is_up_to_date(self, force, rebuild_only, refreshed, current_hash) -> bool:
        if force or rebuild_only:
            return False
        if not (os.path.exists(self.raw_path) and os.path.exists(self.weighted_path)):
            return False
        if any(year < self.year for year in refreshed):
            return False
        meta = self._read_meta()
        return bool(current_hash) and meta.get('sha256') == current_hash

    def _load_contract_bars(self) -> pd.DataFrame:
        frames = []
        for year in self.years():
            path = self.cache_path(year)
            if not os.path.exists(path):
                logger.warning('%s does not exist, skipping ...', path)
                continue
            try:
                parsed = _clean_contract_bars(_read_czce_history(path))
            except (ValueError, KeyError) as exc:
                logger.warning('%s skipped (%s)', path, exc)
                continue
            frames.append(parsed)
        if not frames:
            raise ValueError(
                f'No usable cache files for {self.category}; '
                f'need columns {_REQUIRED_COLUMNS}'
            )
        data = pd.concat(frames, ignore_index=True)
        data = data.drop_duplicates(subset=['date', 'contract'], keep='last')
        return data.sort_values(['date', 'contract']).reset_index(drop=True)

    def _read_meta(self) -> dict:
        if not os.path.exists(self._meta_path):
            return {}
        try:
            with open(self._meta_path, encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_meta(self, current_hash, refreshed) -> None:
        payload = {
            'symbol': self.category,
            'year': self.year,
            'sha256': current_hash,
            'refreshed_years': refreshed,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        os.makedirs(self.cache_dir, exist_ok=True)
        tmp_path = f'{self._meta_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._meta_path)

    @staticmethod
    def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
        return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Download CZCE history and build OI-weighted daily bars.',
    )
    parser.add_argument(
        'symbols',
        nargs='*',
        default=None,
        help='Futures symbols (default: all registered products: %s)'
        % ', '.join(list_products()),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--force', action='store_true', help='Re-download every year')
    mode.add_argument(
        '--rebuild-only',
        action='store_true',
        help='Rebuild CSVs from local cache without downloading',
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    symbols = require_products(args.symbols or list_products())
    failed = []
    for symbol in symbols:
        try:
            DataUpdate(symbol).update(force=args.force, rebuild_only=args.rebuild_only)
        except Exception as exc:
            failed.append((symbol, exc))
            logger.error('%s failed: %s', symbol, exc)
    if failed:
        names = ', '.join(sym for sym, _ in failed)
        raise SystemExit(f'Data update failed for: {names}')


if __name__ == '__main__':
    main()
