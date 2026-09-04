"""Optuna tuning: kick off a study as a job with live progress, list/read its
JSON reports, and evaluate the once-only holdout window.
"""

from __future__ import annotations

import glob
import json
import os

from fastapi import APIRouter, HTTPException

from datafeed.products import require_products
from research.objective import DEFAULT_SPARSE_PENALTY
from research.optimize import RESULTS_DIR as OPTUNA_RESULTS_DIR
from research.optimize import evaluate_holdout, run_study
from research.space import parse_param_override
from strategies import load_strategy

from web import store
from web.jobs import manager as job_manager
from web.marketcache import cache as market_cache
from web.schemas import OptimizeRequest
from web.serialize import jsonable

router = APIRouter(prefix='/api/optimize', tags=['optimize'])


def _optimize_metrics(report: dict) -> dict:
    """A handful of top-level report fields, flattened for the run-history
    list -- not a substitute for the full report (still reachable via
    ``/api/optimize/reports/{name}``), just enough to scan/compare at a glance."""
    diagnostics = report.get('diagnostics', {})
    return {
        'best_value': report.get('best_value'),
        'n_trials_completed': report.get('n_trials_completed'),
        'pbo': diagnostics.get('pbo', {}).get('pbo'),
        'dsr': diagnostics.get('dsr', {}).get('dsr'),
        'is_oos_ratio': diagnostics.get('is_oos_decay', {}).get('ratio'),
    }


@router.post('')
def start_optimize(body: OptimizeRequest):
    try:
        strategy_cls = load_strategy(body.strategy)
        symbols = require_products(body.symbols)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        overrides = dict(parse_param_override(f'{k}={v}') for k, v in body.param_overrides.items())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    run_id = store.create_run(
        kind='optimize', strategy=strategy_cls.__name__, symbols=symbols,
        start=body.start, end=body.end, cash=body.cash, slippage=body.slippage,
        params=overrides,
    )

    def run(job):
        market = market_cache.get(symbols, body.start, body.end)

        # `trial.number` is the study's cumulative count, not this run's --
        # resuming an existing study (no fresh --study-name) starts partway
        # through an existing numbering, which would otherwise show as e.g.
        # "Trial 10/5". Count callback invocations instead for the X/N the
        # user actually asked for; `trial.number` still goes into
        # `progress_data` below, where it is the more useful, absolute axis.
        completed = 0

        def on_trial(study, trial):
            nonlocal completed
            completed += 1
            job.progress = min(0.99, completed / max(1, body.n_trials))
            job.message = f'Trial {completed}/{body.n_trials}  best={study.best_value:.4f}' \
                if study.best_trial is not None else f'Trial {completed}/{body.n_trials}'
            if trial.value is not None:
                job.progress_data.append({'trial': trial.number, 'value': trial.value})
            job_manager.notify(job.id)
            if job.cancel_requested:
                study.stop()

        try:
            out = run_study(
                strategy_cls=strategy_cls,
                symbols=symbols,
                market=market,
                start=body.start,
                end=body.end,
                cash=body.cash,
                slippage=body.slippage,
                n_trials=body.n_trials,
                n_folds=body.n_folds,
                embargo=body.embargo,
                holdout_frac=body.holdout_frac,
                lambda_std=body.lambda_std,
                min_trades_per_year=body.min_trades_per_year,
                dd_cap=body.dd_cap,
                sparse_penalty=body.sparse_penalty if body.sparse_penalty is not None else DEFAULT_SPARSE_PENALTY,
                param_overrides=overrides or None,
                seed=body.seed,
                probe_samples=body.probe_samples,
                study_name=body.study_name,
                callbacks=[on_trial],
            )
        except Exception as exc:
            store.finish_run(run_id, status='error', error=str(exc))
            raise
        job.progress = 1.0
        store.finish_run(
            run_id, status='done', metrics=_optimize_metrics(out['report']),
            artifact={'report_path': out['path'], 'report': out['report']},
        )
        return {'run_id': run_id, 'path': out['path'], 'report': out['report']}

    job = job_manager.submit('optimize', run)
    return {'job_id': job.id, 'run_id': run_id}


@router.get('/reports')
def list_reports():
    paths = sorted(glob.glob(os.path.join(OPTUNA_RESULTS_DIR, '*_best.json')), reverse=True)
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
            'strategy': report.get('strategy'),
            'symbols': report.get('symbols'),
            'timestamp': report.get('timestamp'),
            'best_value': report.get('best_value'),
            'holdout_evaluated': report.get('holdout_evaluated'),
        })
    return out


def _report_path(name: str) -> str:
    if '/' in name or '\\' in name or not name.endswith('.json'):
        raise HTTPException(status_code=400, detail='Invalid report name')
    path = os.path.join(OPTUNA_RESULTS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f'Unknown report {name!r}')
    return path


@router.get('/reports/{name}')
def get_report(name: str):
    path = _report_path(name)
    with open(path, encoding='utf-8') as f:
        return jsonable(json.load(f))


@router.post('/reports/{name}/holdout')
def run_holdout(name: str, force: bool = False):
    """Runs inline rather than as a job: it is one backtest over the holdout
    slice, seconds rather than the minutes a study takes, and the caller
    wants the report back rather than a job id to poll.

    Still registered with the job manager for the duration -- see
    ``JobManager.foreground``. The engine reads costs and roll rules live,
    per fill, so a registry edit landing mid-evaluation would corrupt the one
    number this endpoint exists to produce, and the guard that holds those
    edits off only sees runs the manager knows about.
    """
    path = _report_path(name)
    try:
        with job_manager.foreground('holdout'):
            report = evaluate_holdout(path, force=force)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return jsonable(report)
