"""
Futures history download and OI-weighted aggregation.

Downloading and caching are per-exchange and live in ``datafeed.sources``; this
module owns what is the same everywhere -- OI weighting, atomic CSV writes, and
the freshness bookkeeping that lets a re-run skip a needless rebuild.

Usage (from the repo root):
  python -m datafeed.data_update              # incremental: all registered products
  python -m datafeed.data_update SA CF        # selected products only
  python -m datafeed.data_update --force      # re-download everything
  python -m datafeed.data_update --rebuild-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime

import pandas as pd

from .products import get_product, list_products, normalize_symbol, require_products
from .sources import get_source

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

_WEIGHTED_PRICE_COLS = ['open', 'high', 'low', 'close', 'settle']


def _build_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """OI-weight every contract's bar into one daily series per product.

    A contract row only contributes if *all five* prices are real. Expiring
    contracts routinely print ``open = high = low = 0`` on a day they still
    carry open interest and a trade or two, and every exchange here does it.
    Weighting those in drags the day's open/high/low toward zero while its
    close stays honest, which produced OHLC bars that contradicted themselves
    (``close > high``) -- and silently understated ATR and any high/low stop
    built on them.

    Dropping the whole row rather than masking per column is deliberate: it
    keeps all five prices weighted by the same open interest, so the day's
    bar stays internally consistent. The rows lost this way are tail-end
    contracts whose OI share is a fraction of a percent.
    """
    need = _WEIGHTED_PRICE_COLS + ['oi', 'volume']
    work = df.dropna(subset=need)
    live = (work['oi'] > 0) & (work['volume'] > 0)
    priced = (work[_WEIGHTED_PRICE_COLS] > 0).all(axis=1)
    dropped = int((live & ~priced).sum())
    if dropped:
        logger.info(
            'dropped %d traded contract-day row(s) with a non-positive price '
            'from the OI-weighted series.', dropped,
        )
    work = work[live & priced].copy()
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
    """Write ``df`` to ``path`` in one step, via a temp file in the same
    directory.

    The temp name is unique per call, not a fixed ``f'{path}.tmp'``, for the
    reason ``sources._atomic_path`` already documents: two writers resolving
    the same target open the same temp file, one ``os.replace`` wins, and the
    loser either writes its tail into the winner's already-renamed inode or
    dies on ``FileNotFoundError`` partway through. The web panel de-duplicates
    downloads by product, but only its own -- a ``python ft.py data SA`` in a
    terminal while the panel updates SA is two processes on this exact path.

    Cleanup is a ``finally`` so a failed ``to_csv`` (or a Ctrl-C mid-write)
    leaves nothing behind, and unconditional because a successful
    ``os.replace`` has already moved the temp file away.
    """
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=f'.{os.path.basename(path)}.', suffix='.tmp',
    )
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class DataUpdate:
    def __init__(self, category: str, data_dir: str = None, cache_dir: str = None):
        meta = get_product(category)
        self.category = normalize_symbol(category)
        self.exchange = meta['exchange']
        self.start_year = meta['start_year']
        self.data_dir = data_dir or DATA_DIR
        self.cache_dir = cache_dir or CACHE_DIR
        self.raw_path = os.path.join(self.data_dir, f'{self.category}.csv')
        self.weighted_path = os.path.join(self.data_dir, f'{self.category}_weighted.csv')
        self._meta_path = os.path.join(self.cache_dir, f'{self.category}.meta.json')
        self.source = get_source(self.category, meta, self.cache_dir)
        # Cache units the exchange would not hand over on the last ``update``.
        # Empty means the CSVs are as current as the venue is.
        self.stale_keys: list = []

    def update(self, force: bool = False, rebuild_only: bool = False) -> pd.DataFrame:
        """Sync the exchange cache (incrementally) and rebuild contract / weighted CSVs.

        A download that fails is not fatal -- the existing cache still builds a
        usable series -- but it leaves the result behind the exchange, so the
        keys involved are recorded in ``stale_keys`` for the caller to act on
        rather than being swallowed by a log line.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        refreshed = [] if rebuild_only else self.source.sync(force=force)
        self.stale_keys = list(self.source.failures) if not rebuild_only else []
        if self.stale_keys:
            logger.warning(
                '%s: %d cache unit(s) could not be downloaded (%s%s); '
                'the CSVs below are rebuilt from what is on disk and may be '
                'behind the exchange. Re-run to pick them up.',
                self.category, len(self.stale_keys), ', '.join(self.stale_keys[:5]),
                ', ...' if len(self.stale_keys) > 5 else '',
            )

        fingerprint = self.source.fingerprint()
        if self._is_up_to_date(force, rebuild_only, refreshed, fingerprint):
            logger.info('%s already up to date; skip rebuild.', self.category)
            # Skipping the rebuild does not make this run's download report
            # irrelevant: the meta file is what a later reader consults, so it
            # has to carry *this* run's stale keys, not the previous run's.
            self._write_meta(fingerprint, refreshed)
            return pd.read_csv(self.raw_path, parse_dates=['date'])

        data = self.source.load_bars()
        _to_csv_atomic(self._format_dates(data), self.raw_path)
        weighted = _build_weighted(data)
        _to_csv_atomic(self._format_dates(weighted), self.weighted_path)
        self._write_meta(fingerprint, refreshed)
        logger.info(
            '%s saved %d contract rows / %d weighted days -> %s',
            self.category, len(data), len(weighted), self.data_dir,
        )
        return data

    def _is_up_to_date(self, force, rebuild_only, refreshed, fingerprint) -> bool:
        if force or rebuild_only:
            return False
        if not (os.path.exists(self.raw_path) and os.path.exists(self.weighted_path)):
            return False
        # The fingerprint only covers the newest cache unit, so refreshing
        # anything older always forces a rebuild.
        head = self.source.head_key()
        if any(key != head for key in refreshed):
            return False
        meta = self._read_meta()
        return bool(fingerprint) and meta.get('fingerprint') == fingerprint

    def _read_meta(self) -> dict:
        if not os.path.exists(self._meta_path):
            return {}
        try:
            with open(self._meta_path, encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_meta(self, fingerprint, refreshed) -> None:
        payload = {
            'symbol': self.category,
            'exchange': self.exchange,
            'head': self.source.head_key(),
            'fingerprint': fingerprint,
            'refreshed': refreshed,
            # Non-empty means this build is known-incomplete: whoever reads the
            # CSVs later can see that without re-running the download.
            'stale_keys': list(self.stale_keys),
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        os.makedirs(self.cache_dir, exist_ok=True)
        # Unique temp name and a cleaning `finally`, same as `_to_csv_atomic`
        # above and for the same collision.
        fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_dir, prefix=f'.{os.path.basename(self._meta_path)}.', suffix='.tmp',
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._meta_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['date'] = pd.to_datetime(out['date']).dt.strftime('%Y-%m-%d')
        return out


def run_updates(symbols=None, *, force: bool = False, rebuild_only: bool = False) -> None:
    """Refresh every symbol in ``symbols`` (default: every registered product).

    The one implementation behind both entry points -- ``ft.py data`` and
    ``python -m datafeed.data_update`` -- so the partial-failure contract
    below is stated once. Raises ``SystemExit`` if any product failed or came
    back stale.
    """
    symbols = require_products(symbols or list_products())
    failed = []
    stale = []
    for symbol in symbols:
        # Construction can fail too -- an unsupported exchange, a malformed
        # registry row -- and that is this product's problem, not the run's.
        job = None
        try:
            job = DataUpdate(symbol)
            job.update(force=force, rebuild_only=rebuild_only)
        except Exception as exc:
            failed.append((symbol, exc))
            logger.error('%s failed: %s', symbol, exc)
            continue
        if job is not None and job.stale_keys:
            stale.append((symbol, len(job.stale_keys)))

    # A partial download is the dangerous outcome: the CSVs exist, look normal,
    # and are quietly behind the exchange. Exit non-zero so a scheduled refresh
    # cannot report success on data it failed to fetch.
    problems = []
    if failed:
        problems.append('failed for: ' + ', '.join(sym for sym, _ in failed))
    if stale:
        problems.append(
            'incomplete (stale, re-run to complete): '
            + ', '.join(f'{sym} ({n})' for sym, n in stale)
        )
    if problems:
        raise SystemExit('Data update ' + '; '.join(problems))


def add_cli_args(parser) -> None:
    """Declare the download flags on ``parser``.

    Shared with ``ft.py`` so the two entry points cannot drift apart on flag
    names or help text.
    """
    parser.add_argument(
        'symbols',
        nargs='*',
        default=None,
        help='Futures symbols (default: all registered products: %s)'
        % ', '.join(list_products()),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--force',
        action='store_true',
        help='Re-download everything. Cheap for CZCE (a file per year), but for '
             'SHFE/DCE this re-fetches every trading day since start_year.',
    )
    mode.add_argument(
        '--rebuild-only',
        action='store_true',
        help='Rebuild CSVs from local cache without downloading',
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Download exchange history and build OI-weighted daily bars.',
    )
    add_cli_args(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_updates(args.symbols, force=args.force, rebuild_only=args.rebuild_only)


if __name__ == '__main__':
    main()
