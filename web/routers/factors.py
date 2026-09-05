"""Factor evaluation: judge a predictive score directly (IC, quantile
buckets, decay, turnover, cross-factor correlation) as a background job,
then list/read its JSON reports.

Thin wrapper over ``research.factor_report`` -- the same
``build_report``/``build_corr_report``/``save_report`` functions
``factor_runner.py``'s ``report``/``corr`` subcommands use, so a report
saved from the CLI and one saved from this panel land in the same
``results/factors/`` directory, in the same shape, and list together below.
"""

from __future__ import annotations

import glob
import inspect
import json
import os

from fastapi import APIRouter, HTTPException

from datafeed.products import require_products
from factors import discover_factors, load_factor
from research.factor_report import (
    RESULTS_DIR as FACTOR_RESULTS_DIR,
    build_corr_report, build_factor_panel, build_report, save_report,
)
from research.space import resolve_space, spec_to_json

from web.jobs import manager as job_manager
from web.marketcache import cache as market_cache
from web.schemas import FactorCorrRequest, FactorReportRequest
from web.serialize import jsonable

router = APIRouter(prefix='/api/factors', tags=['factors'])


def _describe(key: str, cls: type) -> dict:
    try:
        space = {k: spec_to_json(v) for k, v in resolve_space(cls).items()}
        space_error = None
    except ValueError as e:
        space, space_error = {}, str(e)

    return {
        'key': key,
        'class_name': cls.__name__,
        'module': cls.__module__,
        'docstring': inspect.getdoc(cls) or '',
        'direction': getattr(cls, 'direction', 1),
        'params': dict(getattr(cls, 'params', {}) or {}),
        'fixed_params': list(getattr(cls, 'fixed_params', ()) or ()),
        'space': space,
        'space_error': space_error,
    }


@router.get('')
def list_factors():
    return [_describe(key, cls) for key, cls in sorted(discover_factors().items())]


@router.post('/report')
def start_report(body: FactorReportRequest):
    try:
        symbols = require_products(body.symbols)
        factor_cls = load_factor(body.factor)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        factor = factor_cls(**body.params)
    except TypeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    def run(job):
        job.message = 'Loading market data'
        market = market_cache.get(symbols, body.start, body.end)
        job.progress = 0.2
        job.message = f'Scoring {factor_cls.__name__}'
        report = build_report(
            market, factor, horizons=body.horizons, n_groups=body.n_groups,
            return_source=body.return_source,
        )
        job.progress = 0.9
        payload = {**report, 'start': body.start, 'end': body.end}
        path = save_report(payload, f'{body.factor}_report', FACTOR_RESULTS_DIR)
        job.progress = 1.0
        return {'path': path, 'report': payload}

    job = job_manager.submit('factor-report', run)
    return {'job_id': job.id}


@router.post('/corr')
def start_corr(body: FactorCorrRequest):
    try:
        symbols = require_products(body.symbols)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    registry = discover_factors()
    if len(registry) < 2:
        raise HTTPException(
            status_code=422,
            detail=f'corr needs at least 2 discovered factors under factors/; found {sorted(registry)}',
        )
    names = sorted(registry)

    def run(job):
        job.message = 'Loading market data'
        market = market_cache.get(symbols, body.start, body.end)
        panels = {}
        for i, name in enumerate(names):
            job.message = f'Scoring {name}'
            panels[name] = build_factor_panel(registry[name](), market)
            job.progress = 0.1 + 0.7 * (i + 1) / len(names)
        result = build_corr_report(market, panels, horizon=body.horizon)
        job.progress = 0.95
        payload = {**result, 'start': body.start, 'end': body.end}
        path = save_report(payload, 'corr', FACTOR_RESULTS_DIR)
        job.progress = 1.0
        return {'path': path, 'report': payload}

    job = job_manager.submit('factor-corr', run)
    return {'job_id': job.id}


@router.get('/reports')
def list_reports():
    paths = sorted(glob.glob(os.path.join(FACTOR_RESULTS_DIR, '*_report_*.json')), reverse=True)
    out = []
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path, encoding='utf-8') as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            'name': name,
            'factor': report.get('factor'),
            'symbols': report.get('symbols'),
            'start': report.get('start'),
            'end': report.get('end'),
            'n_groups': report.get('n_groups'),
            'return_source': report.get('return_source'),
        })
    return out


def _report_path(name: str) -> str:
    if '/' in name or '\\' in name or not name.endswith('.json'):
        raise HTTPException(status_code=400, detail='Invalid report name')
    path = os.path.join(FACTOR_RESULTS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f'Unknown report {name!r}')
    return path


@router.get('/reports/{name}')
def get_report(name: str):
    path = _report_path(name)
    with open(path, encoding='utf-8') as f:
        return jsonable(json.load(f))


# Registered last: a bare ``{key}`` wildcard would otherwise swallow the
# more specific ``/report``, ``/reports``, ``/reports/{name}`` and ``/corr``
# routes above it, since FastAPI/Starlette matches path *and* method in
# registration order and every one of those is also a single path segment
# under this router's prefix.
@router.get('/{key}')
def get_factor(key: str):
    try:
        cls = load_factor(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _describe(key, cls)
