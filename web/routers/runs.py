"""Run history: list/get/compare/delete over ``web.store``'s SQLite index.

Equity curves and trade logs live in the per-run JSON artifact
(``results/web/{id}.json``), not in a response model here -- see
``web/store.py`` for why they are split out of the index.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from web import store
from web.serialize import annotate_inf_metrics, jsonable

router = APIRouter(prefix='/api/runs', tags=['runs'])


def _run_summary(row: dict) -> dict:
    out = dict(row)
    if out.get('metrics'):
        out['metrics'] = annotate_inf_metrics(out['metrics'])
    return jsonable(out)


@router.get('')
def list_runs(kind: str = None, limit: int = 100):
    return [_run_summary(r) for r in store.list_runs(kind=kind, limit=limit)]


@router.get('/compare')
def compare_runs(ids: str = Query(..., description='Comma-separated run ids')):
    run_ids = [i.strip() for i in ids.split(',') if i.strip()]
    rows = []
    curves = {}
    for run_id in run_ids:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f'Unknown run {run_id!r}')
        rows.append(_run_summary(row))
        artifact = store.get_artifact(run_id)
        if artifact and artifact.get('equity_records'):
            curves[run_id] = [
                {'date': str(r['date']), 'equity': r['equity']}
                for r in artifact['equity_records']
            ]
    return jsonable({'runs': rows, 'equity_curves': curves})


@router.get('/{run_id}')
def get_run(run_id: str):
    row = store.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f'Unknown run {run_id!r}')
    artifact = store.get_artifact(run_id) or {}
    out = _run_summary(row)
    out['equity_records'] = jsonable(artifact.get('equity_records', []))
    out['trade_logs'] = jsonable(artifact.get('trade_logs', []))
    out['deferred'] = jsonable(artifact.get('deferred', {}))
    out['symbols_with_price'] = sorted((artifact.get('price') or {}).keys())
    return out


@router.get('/{run_id}/price/{symbol}')
def get_run_price(run_id: str, symbol: str):
    row = store.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f'Unknown run {run_id!r}')
    artifact = store.get_artifact(run_id) or {}
    price = (artifact.get('price') or {}).get(symbol.upper())
    if price is None:
        raise HTTPException(status_code=404, detail=f'No price data for {symbol!r} in run {run_id!r}')
    signals = (artifact.get('signal_log_by_symbol') or {}).get(symbol.upper(), [])
    return jsonable({'symbol': symbol.upper(), **price, 'signals': signals})


@router.delete('/{run_id}')
def delete_run(run_id: str):
    ok = store.delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f'Unknown run {run_id!r}')
    return {'deleted': run_id}
