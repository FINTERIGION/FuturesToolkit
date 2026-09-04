"""Data download manager: coverage table + a job that runs
``datafeed.data_update.DataUpdate`` per symbol.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException

from datafeed.data_update import DataUpdate
from datafeed.products import list_products, normalize_symbol, require_products

from web.data_status import coverage_for
from web.jobs import manager as job_manager
from web.marketcache import cache as market_cache
from web.schemas import DataUpdateRequest
from web.serialize import jsonable

router = APIRouter(prefix='/api/data', tags=['data'])
logger = logging.getLogger('futurestoolkit.web')

# Symbols with a download in flight. Two updates of the same product write the
# same `{sym}.csv` / `{sym}_weighted.csv` / `{sym}.meta.json`: each file lands
# atomically on its own, but the pair is fetched and rebuilt independently, so
# the loser overwrites the winner with a frame built from a different fetch.
# The pool runs two jobs at once, so clicking Update twice is enough to hit it.
_inflight: set = set()
_inflight_lock = threading.Lock()


@router.get('/coverage')
def coverage():
    return jsonable([coverage_for(code) for code in list_products()])


@router.post('/update')
def start_update(body: DataUpdateRequest):
    try:
        symbols = require_products(body.symbols) if body.symbols else list_products()
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Claim the symbols before submitting, so the check and the claim cannot
    # be interleaved by a second request arriving between them.
    with _inflight_lock:
        clash = sorted(set(symbols) & _inflight)
        if clash:
            raise HTTPException(
                status_code=409,
                detail=f'Already downloading {", ".join(clash)}; wait for that job '
                       f'to finish or cancel it before starting another for the '
                       f'same product.',
            )
        _inflight.update(symbols)

    def run(job):
        n = len(symbols)
        stale = []
        try:
            for i, sym in enumerate(symbols):
                if job.cancel_requested:
                    job.message = f'Cancelled before {sym}'
                    break
                job.message = f'Updating {sym} ({i + 1}/{n})'
                job.progress = i / n
                try:
                    update = DataUpdate(normalize_symbol(sym))
                    update.update(force=body.force, rebuild_only=body.rebuild_only)
                    if update.stale_keys:
                        stale.append(sym)
                except Exception as exc:
                    logger.error('Data update failed for %s: %s', sym, exc)
                    raise
        finally:
            # Release on every exit -- a raise or a cancel must not leave a
            # symbol permanently unclaimable for the life of the process.
            with _inflight_lock:
                _inflight.difference_update(symbols)
        job.progress = 1.0
        market_cache.invalidate()
        return {'symbols': symbols, 'stale': stale}

    try:
        job = job_manager.submit('data_update', run)
    except Exception:
        with _inflight_lock:
            _inflight.difference_update(symbols)
        raise
    return {'job_id': job.id}
