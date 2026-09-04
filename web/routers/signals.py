"""Live signals: next session's target positions, for a plain strategy or a
meta-gated one -- see ``live/signal.py`` for the mechanism and its caveats
(``current_simulated`` is not your book; equity-sized strategies need your
real account equity passed as ``cash``).
"""

from __future__ import annotations

import glob
import json
import os

from fastapi import APIRouter, HTTPException

from datafeed.products import require_products
from research.runner_api import load_market
from runner import resolve_params
from strategies import load_strategy

from live.signal import SignalSpec, compute_signal, spec_from_model
from web.config import LIVE_RESULTS_DIR, resolve_model_path
from web.jobs import manager as job_manager
from web.schemas import SignalRequest
from web.serialize import jsonable

router = APIRouter(prefix='/api/signals', tags=['signals'])


def _build_spec(body: SignalRequest) -> SignalSpec:
    if bool(body.model) == bool(body.strategy):
        raise ValueError('Exactly one of `strategy` or `model` must be set')

    if body.model:
        # ValueError from an out-of-tree path lands on `start_signal`'s own
        # 422 handler, alongside the other request-shape failures.
        model_path = resolve_model_path(body.model)
        spec = spec_from_model(model_path, symbols=body.symbols, cash=body.cash, slippage=body.slippage)
        return SignalSpec(**{**vars(spec), 'symbols': require_products(spec.symbols)})

    strategy_cls = load_strategy(body.strategy)
    return SignalSpec(
        strategy_cls=strategy_cls,
        params=dict(body.params or {}),
        symbols=require_products(body.symbols or ['SA', 'FG', 'CF', 'C']),
        cash=100_000.0 if body.cash is None else body.cash,
        slippage=0.0 if body.slippage is None else body.slippage,
    )


@router.post('')
def start_signal(body: SignalRequest):
    try:
        spec = _build_spec(body)
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    def run(job):
        job.message = 'Loading market data'
        market = load_market(spec.symbols, body.start, body.end)
        job.progress = 0.5
        job.message = 'Replaying to the last bar'
        report = compute_signal(market, spec)
        job.progress = 1.0
        os.makedirs(LIVE_RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(
            LIVE_RESULTS_DIR, f'signal_{spec.strategy_name}_{report.as_of}.json',
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        return report.to_dict()

    job = job_manager.submit('signal', run)
    return {'job_id': job.id}


@router.get('/history')
def signal_history():
    paths = sorted(glob.glob(os.path.join(LIVE_RESULTS_DIR, 'signal_*.json')), reverse=True)
    out = []
    for path in paths:
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            'file': os.path.basename(path),
            'as_of': data.get('as_of'),
            'strategy': data.get('strategy'),
            'symbols': data.get('symbols'),
        })
    return out


@router.get('/history/{file}')
def signal_detail(file: str):
    if '/' in file or '\\' in file or not file.endswith('.json'):
        raise HTTPException(status_code=400, detail='Invalid file name')
    path = os.path.join(LIVE_RESULTS_DIR, file)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f'Unknown file {file!r}')
    with open(path, encoding='utf-8') as f:
        return jsonable(json.load(f))
