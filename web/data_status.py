"""Lightweight, on-disk-only data coverage for one registered product.

Deliberately does not touch ``DataManager``/``MarketData`` -- building a full
universe bundle just to answer "how many rows and how fresh" for a coverage
table would mean loading every product's roll calendar and contract frames
merely to display a row count. This reads the CSV header/index and the cache
meta file directly, the same two files ``DataManager`` and ``DataUpdate``
already produce.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

from datafeed.data_manager import DataManager
from datafeed.data_update import CACHE_DIR

_DATA_DIR = DataManager.DATA_DIR


def coverage_for(symbol: str) -> dict:
    weighted_path = os.path.join(_DATA_DIR, f'{symbol}_weighted.csv')
    meta_path = os.path.join(CACHE_DIR, f'{symbol}.meta.json')

    info = {
        'symbol': symbol,
        'has_data': False,
        'n_rows': 0,
        'first_date': None,
        'last_date': None,
        'last_refresh': None,
        'stale_keys': [],
    }

    if os.path.exists(weighted_path):
        try:
            dates = pd.read_csv(weighted_path, usecols=['date'])['date']
        except (ValueError, pd.errors.EmptyDataError):
            dates = pd.Series([], dtype=str)
        if len(dates):
            info['has_data'] = True
            info['n_rows'] = int(len(dates))
            info['first_date'] = str(dates.min())
            info['last_date'] = str(dates.max())

    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            info['last_refresh'] = meta.get('updated_at')
            info['stale_keys'] = meta.get('stale_keys', [])
        except (json.JSONDecodeError, OSError):
            pass

    return info


def data_file_paths(symbol: str) -> list:
    """Every on-disk file a product's data occupies -- used by the "purge
    data" option on delete. Cache payloads (``cache/{CZCE,SHFE,DCE}/...``)
    are shared across products on SHFE/DCE and are intentionally left
    alone; only this product's own generated CSVs and meta file are listed.
    """
    paths = [
        os.path.join(_DATA_DIR, f'{symbol}.csv'),
        os.path.join(_DATA_DIR, f'{symbol}_weighted.csv'),
        os.path.join(CACHE_DIR, f'{symbol}.meta.json'),
    ]
    return [p for p in paths if os.path.exists(p)]
