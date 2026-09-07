"""Product registry CRUD, price data for the chart, and roll-calendar
lookup. See ``datafeed/products.py`` for the underlying registry and
``web/data_status.py`` for the on-disk coverage helper.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query

from datafeed.data_manager import DataManager
from datafeed.products import (
    DEFAULT_MAIN_MONTHS,
    DEFAULT_ROLL_LEAD_MONTHS,
    PRODUCTS,
    SUPPORTED_EXCHANGES,
    get_product,
    list_products,
    normalize_symbol,
    product_costs,
    roll_rule,
    save_registry,
)

from web.data_status import coverage_for, data_file_paths
from web.jobs import manager as job_manager
from web.marketcache import cache as market_cache
from web.schemas import ProductIn
from web.serialize import jsonable

router = APIRouter(prefix='/api', tags=['products'])


def _write_registry(registry: dict) -> None:
    """Persist the registry, but only while nothing is mid-run.

    The check and the write happen inside ``hold_starts`` so no job can begin
    between them: the engine reads ``product_costs``/``roll_rule`` live, per
    fill (``core/broker.py``, ``core/ledger.py``), so a multiplier or margin
    edit landing on a run already in flight silently corrupts that run's
    numbers rather than failing.
    """
    with job_manager.hold_starts():
        if job_manager.any_active():
            raise HTTPException(
                status_code=409,
                detail='A job is running. Product edits are held while a backtest, '
                       'optimize, data update, or holdout evaluation is in progress, '
                       'because the engine reads costs and roll rules live, per fill.',
            )
        try:
            save_registry(registry)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
    market_cache.invalidate()


def _product_view(code: str) -> dict:
    meta = get_product(code)
    return {
        'code': code,
        **meta,
        'costs': product_costs(code),
        'roll': roll_rule(code),
        'coverage': coverage_for(code),
    }


@router.get('/products')
def list_all_products():
    return [_product_view(code) for code in list_products()]


@router.get('/products/{code}')
def get_one_product(code: str):
    code = normalize_symbol(code)
    if code not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f'Unknown product {code!r}')
    return _product_view(code)


@router.post('/products/{code}', status_code=201)
def create_product(code: str, body: ProductIn):
    """``code`` in the path, matching PUT and DELETE.

    It used to be a query parameter here alone -- the same identifier
    addressed two different ways depending on the verb.
    """
    code = normalize_symbol(code)
    if code in PRODUCTS:
        raise HTTPException(status_code=409, detail=f'{code} already exists; use PUT to edit it')

    registry = dict(PRODUCTS)
    registry[code] = body.to_registry_entry()
    _write_registry(registry)
    return _product_view(code)


@router.put('/products/{code}')
def update_product(code: str, body: ProductIn):
    code = normalize_symbol(code)
    if code not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f'Unknown product {code!r}')

    registry = dict(PRODUCTS)
    registry[code] = body.to_registry_entry()
    _write_registry(registry)
    return _product_view(code)


@router.delete('/products/{code}')
def delete_product(code: str, purge_data: bool = Query(False)):
    code = normalize_symbol(code)
    if code not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f'Unknown product {code!r}')

    registry = dict(PRODUCTS)
    del registry[code]
    _write_registry(registry)

    removed = []
    if purge_data:
        for path in data_file_paths(code):
            os.remove(path)
            removed.append(path)

    return {'deleted': code, 'purged_files': removed}


@router.get('/products/{code}/bars')
def product_bars(code: str, start: str = None, end: str = None):
    code = normalize_symbol(code)
    if code not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f'Unknown product {code!r}')
    try:
        dm = DataManager(symbols=[code])
        df = dm.load_dataframe(start_date=start, end_date=end, symbol=code)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    rows = [
        {'date': str(idx.date()), **{c: row[c] for c in df.columns}}
        for idx, row in df.iterrows()
    ]
    return jsonable({'symbol': code, 'bars': rows})


@router.get('/products/{code}/roll')
def product_roll(code: str, start: str = None, end: str = None):
    code = normalize_symbol(code)
    if code not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f'Unknown product {code!r}')
    try:
        dm = DataManager(symbols=[code])
        bundle = dm.get_universe_bundle(start_date=start, end_date=end)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    exec_df = bundle['products'][code]['exec_price_df']
    rows = [
        {'date': str(idx.date()), 'contract': row['contract'], 'close': row['close']}
        for idx, row in exec_df.iterrows()
        if row['contract']
    ]
    return jsonable({'symbol': code, 'roll': rows})


@router.get('/meta/exchanges')
def meta_exchanges():
    return {
        'exchanges': list(SUPPORTED_EXCHANGES),
        'default_main_months': list(DEFAULT_MAIN_MONTHS),
        'default_roll_lead_months': DEFAULT_ROLL_LEAD_MONTHS,
    }
