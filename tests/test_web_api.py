"""Web panel backend: JSON serialization safety, product registry CRUD
through the API, the job manager's lifecycle, and one real end-to-end
backtest run through the HTTP surface.

Two isolation fixtures below matter more than they look: without them, a
test posting to ``/api/products`` would overwrite the repo's real
``datafeed/products.json`` and leave every other test in the session (and
the working tree) looking at throwaway data. See each fixture's docstring
for the mechanism -- it relies on how ``save_registry``/``web.store``
resolve their default paths as module globals at *call* time, not at
function-definition time, which is what makes monkeypatching the module
attribute (rather than passing an explicit path through five layers of
router code) actually take effect.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

import datafeed.products as products_module
import web.jobs as jobs_module
import web.marketcache as marketcache_module
import web.store as store_module
from tests.conftest import build_trending_market
from web.app import app
from web.config import STATIC_DIR
from web.jobs import JobManager
from web.schemas import BacktestRequest, DataUpdateRequest, OptimizeRequest
from web.serialize import annotate_inf_metrics, jsonable

_VALID_PRODUCT = {
    'exchange': 'CZCE',
    'name': 'test_product',
    'name_zh': '测试品种',
    'start_year': 2020,
    'main_months': [1, 5, 9],
    'multiplier': 10,
    'tick_size': 1.0,
    'margin_rate': 0.12,
    'commission_rate': 0.0001,
}


@pytest.fixture
def client():
    """``base_url`` names a host the app will actually answer to.

    ``TestClient``'s own default is ``http://testserver``, and the app refuses
    a Host header outside ``web.config.allowed_hosts`` -- so leaving it at the
    default would have every test in this module exercise the rejection path
    rather than the endpoint it means to. A browser reaching the panel sends
    one of these names too, so this is the realistic header, not a workaround.
    """
    return TestClient(app, base_url='http://localhost')


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Redirect every ``save_registry``/``reload_registry`` call made during
    a test at a throwaway copy of the registry, and restore the real
    in-memory ``PRODUCTS`` afterward.

    ``save_registry`` resolves its default write path as ``path or
    REGISTRY_PATH``, with ``REGISTRY_PATH`` looked up as a bare name in
    ``datafeed.products``'s own globals at call time -- so monkeypatching
    the module attribute here redirects every caller (the products router
    included), regardless of how each imported the function.
    """
    registry_path = tmp_path / 'products.json'
    registry_path.write_text(
        json.dumps({code: dict(meta) for code, meta in products_module.PRODUCTS.items()}),
        encoding='utf-8',
    )
    monkeypatch.setattr(products_module, 'REGISTRY_PATH', str(registry_path))

    snapshot = {code: dict(meta) for code, meta in products_module.PRODUCTS.items()}
    order = list(products_module.PRODUCTS)
    yield
    products_module.PRODUCTS.clear()
    products_module.PRODUCTS.update({code: snapshot[code] for code in order})


@pytest.fixture(autouse=True)
def synthetic_market(monkeypatch):
    """Serve every run in this module a synthetic market instead of reading
    `data/*.csv`.

    Those CSVs are gitignored, so they exist on the machine that downloaded
    them and nowhere else -- a fresh clone and CI both have an empty `data/`,
    which used to make the two end-to-end tests below pass locally and fail
    on checkout. What they actually cover is the HTTP surface and the
    job/store/serialization plumbing above the engine, none of which cares
    whether the bars are real; the feed itself is covered by
    `test_data_update` and `test_sources`.

    Patched on the cache singleton the routers already hold a reference to,
    which is the one seam every run type funnels through.
    """
    built = {}

    def fake_get(symbols, start, end, update=False):
        key = tuple(sorted(symbols))
        if key not in built:
            built[key] = build_trending_market(key)
        return built[key]

    monkeypatch.setattr(marketcache_module.cache, 'get', fake_get)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the run-history SQLite index and artifact directory at a tmp
    path, and drop the cached connection so the next call reconnects there.
    """
    monkeypatch.setattr(store_module, 'DB_PATH', str(tmp_path / 'webpanel_test.db'))
    monkeypatch.setattr(store_module, 'WEB_RESULTS_DIR', str(tmp_path / 'web_results'))
    monkeypatch.setattr(store_module, '_conn', None)
    yield
    monkeypatch.setattr(store_module, '_conn', None)


# ---------------------------------------------------------------------
# Serialization safety
# ---------------------------------------------------------------------

def test_jsonable_produces_valid_json_for_inf_nan_dates_and_numpy():
    import datetime

    import numpy as np

    payload = {
        'inf': float('inf'), 'neg_inf': float('-inf'), 'nan': float('nan'),
        'date': datetime.date(2024, 1, 1),
        'np_float': np.float64(1.5), 'np_int': np.int64(3), 'np_arr': np.array([1, 2]),
        'nested': [1, {'x': float('nan')}],
    }
    text = json.dumps(jsonable(payload))       # must not raise, must be strict-JSON
    back = json.loads(text)
    assert back['inf'] is None
    assert back['neg_inf'] is None
    assert back['nan'] is None
    assert back['date'] == '2024-01-01'
    assert back['np_float'] == 1.5
    assert back['np_arr'] == [1, 2]


def test_annotate_inf_metrics_flags_positive_infinity_only():
    metrics = {'calmar_ratio': float('inf'), 'profit_factor': 2.0, 'profit_loss_ratio': float('-inf')}
    out = annotate_inf_metrics(metrics)
    json.dumps(out)  # must be strict-JSON-safe
    assert out['calmar_ratio'] is None and out['calmar_ratio_is_inf'] is True
    assert out['profit_factor'] == 2.0 and out['profit_factor_is_inf'] is False
    assert out['profit_loss_ratio'] is None and out['profit_loss_ratio_is_inf'] is False  # -inf, not +inf


# ---------------------------------------------------------------------
# Product registry CRUD through the API
# ---------------------------------------------------------------------

def test_list_products_matches_registry(client):
    resp = client.get('/api/products')
    assert resp.status_code == 200
    codes = {row['code'] for row in resp.json()}
    assert codes == set(products_module.PRODUCTS)


def test_create_get_update_delete_product_round_trip(client):
    resp = client.post('/api/products/ZZ', json=_VALID_PRODUCT)
    assert resp.status_code == 201, resp.text
    assert resp.json()['code'] == 'ZZ'

    resp = client.get('/api/products/ZZ')
    assert resp.status_code == 200
    assert resp.json()['name_zh'] == '测试品种'

    updated = dict(_VALID_PRODUCT, margin_rate=0.20)
    resp = client.put('/api/products/ZZ', json=updated)
    assert resp.status_code == 200
    assert resp.json()['margin_rate'] == pytest.approx(0.20)

    resp = client.delete('/api/products/ZZ')
    assert resp.status_code == 200
    assert resp.json()['deleted'] == 'ZZ'

    resp = client.get('/api/products/ZZ')
    assert resp.status_code == 404


def test_create_duplicate_product_is_rejected(client):
    resp = client.post('/api/products/SA', json=_VALID_PRODUCT)
    assert resp.status_code == 409


def test_create_product_with_bad_tick_size_is_422(client):
    bad = dict(_VALID_PRODUCT, tick_size=-1)
    resp = client.post('/api/products/YY', json=bad)
    assert resp.status_code == 422
    assert 'tick_size' in resp.json()['detail']


def test_create_product_with_neither_commission_mode_is_422(client):
    bad = dict(_VALID_PRODUCT)
    bad.pop('commission_rate')
    resp = client.post('/api/products/YY', json=bad)
    assert resp.status_code == 422


def test_update_unknown_product_is_404(client):
    resp = client.put('/api/products/NOPE', json=_VALID_PRODUCT)
    assert resp.status_code == 404


def test_product_write_is_refused_while_a_job_is_running(client):
    import threading

    from web.jobs import manager as job_manager

    release = threading.Event()

    def blocking(job):
        release.wait(timeout=5)
        return None

    job = job_manager.submit('test_lock', blocking)
    try:
        for _ in range(50):
            if job.status == 'running':
                break
            time.sleep(0.02)
        assert job.status == 'running'

        resp = client.post('/api/products/YY', json=_VALID_PRODUCT)
        assert resp.status_code == 409
    finally:
        release.set()
        for _ in range(50):
            if job.status != 'running':
                break
            time.sleep(0.02)


# ---------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------

def test_job_manager_runs_to_completion_and_reports_result():
    mgr = JobManager(max_workers=1)
    job = mgr.submit('unit', lambda j: {'ok': True})
    for _ in range(100):
        if job.status in ('done', 'error'):
            break
        time.sleep(0.02)
    assert job.status == 'done'
    assert job.result == {'ok': True}


def test_job_manager_captures_exceptions_as_error_status():
    mgr = JobManager(max_workers=1)

    def boom(job):
        raise ValueError('kaboom')

    job = mgr.submit('unit', boom)
    for _ in range(100):
        if job.status in ('done', 'error'):
            break
        time.sleep(0.02)
    assert job.status == 'error'
    assert 'kaboom' in job.error


def test_job_manager_cancel_sets_flags_and_rejects_unknown_job():
    import threading

    mgr = JobManager(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def blocking(job):
        started.set()
        release.wait(timeout=5)
        return job.cancel_requested

    job = mgr.submit('unit', blocking)
    assert started.wait(timeout=2)
    assert mgr.cancel(job.id) is True
    assert job.cancel_requested is True
    release.set()

    for _ in range(100):
        if job.status in ('done', 'error', 'cancelled'):
            break
        time.sleep(0.02)
    assert job.result is True  # the callable observed cancel_requested

    assert mgr.cancel('unknown-job-id') is False


def test_get_unknown_job_via_api_is_404(client):
    resp = client.get('/api/jobs/does-not-exist')
    assert resp.status_code == 404


# ---------------------------------------------------------------------
# End-to-end: one real backtest through the HTTP surface
# ---------------------------------------------------------------------

def test_backtest_end_to_end_through_the_api(client):
    resp = client.post('/api/backtest', json={
        'strategy': 'double_ma',
        'symbols': ['SA'],
        'start': '2024-01-01',
        'end': '2024-06-01',
        'cash': 200_000.0,
        'slippage': 0.0,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    job_id, run_id = body['job_id'], body['run_id']

    job = None
    for _ in range(200):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] in ('done', 'error'):
            break
        time.sleep(0.1)
    assert job['status'] == 'done', job

    text = json.dumps(job)               # the whole job payload must be strict JSON
    json.loads(text)
    metrics = job['result']['metrics']
    assert 'sharpe_ratio' in metrics
    assert metrics['calmar_ratio_is_inf'] in (True, False)

    run = client.get(f'/api/runs/{run_id}').json()
    assert run['status'] == 'done'
    assert isinstance(run['equity_records'], list) and len(run['equity_records']) > 0
    assert 'SA' in run['symbols_with_price']

    price = client.get(f'/api/runs/{run_id}/price/SA').json()
    assert len(price['dates']) == len(price['close']) > 0

    listing = client.get('/api/runs').json()
    assert any(r['id'] == run_id for r in listing)

    resp = client.delete(f'/api/runs/{run_id}')
    assert resp.status_code == 200
    assert client.get(f'/api/runs/{run_id}').status_code == 404


def test_optimize_run_is_persisted_into_run_history(client, tmp_path, monkeypatch):
    import research.optimize as optimize_module

    # Match tests/test_research.py's own isolation: run_study's default
    # out_dir is `results_dir or RESULTS_DIR`, resolved as a module global at
    # call time, so this keeps the study's SQLite db and *_best.json report
    # out of the repo's real results/optuna/ directory.
    monkeypatch.setattr(optimize_module, 'RESULTS_DIR', str(tmp_path))

    resp = client.post('/api/optimize', json={
        'strategy': 'double_ma',
        'symbols': ['SA', 'CF'],
        'start': '2020-01-01',
        'end': '2022-01-01',
        'cash': 100_000.0,
        'slippage': 0.0,
        'n_trials': 3,
        'n_folds': 2,
        'embargo': 10,
        'holdout_frac': 0.2,
        'lambda_std': 0.5,
        'min_trades_per_year': 4.0,
        'dd_cap': 0.35,
        'param_overrides': {},
        'seed': 1,
        'probe_samples': 2,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    job_id, run_id = body['job_id'], body['run_id']

    job = None
    for _ in range(200):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['status'] in ('done', 'error'):
            break
        time.sleep(0.1)
    assert job['status'] == 'done', job
    json.loads(json.dumps(job))  # strict-JSON safety, same as the backtest test

    # The panel's "Send to Backtest" reproduces a report by re-running its
    # params under the costs it was tuned with, so these two travel with it and
    # `OptimizeReport` in webui/src/api/types.ts declares them non-optional.
    # Dropping them here would leave that hand-off silently sending undefined.
    report = job['result']['report']
    assert report['cash'] == 100_000.0
    assert report['slippage'] == 0.0
    assert report['strategy_key'] == 'double_ma'

    run = client.get(f'/api/runs/{run_id}').json()
    assert run['kind'] == 'optimize'
    assert run['status'] == 'done'
    assert run['strategy'] == 'DoubleMaStrategy'
    assert 'best_value' in run['metrics']

    listing = client.get('/api/runs?kind=optimize').json()
    assert any(r['id'] == run_id for r in listing)

    resp = client.delete(f'/api/runs/{run_id}')
    assert resp.status_code == 200


def test_optimize_writes_a_terminal_status_when_market_loading_fails(client, monkeypatch):
    """The run row is inserted as ``running`` before the job starts, so every
    exit path owes it a terminal status.

    ``market_cache.get`` sits *before* the study call and used to be outside
    the router's error handling, so a failure there -- an undownloaded symbol,
    a malformed date -- left the row at ``running`` forever:
    ``store._prune_locked`` never deletes a running row, and the panel lists
    only ``kind='backtest'``, so the orphan was unreachable as well as
    immortal. The backtest router has always wrapped its whole callable; this
    asserts optimize now does too.
    """
    def boom(symbols, start, end, update=False):
        raise ValueError('no data for SA')

    monkeypatch.setattr(marketcache_module.cache, 'get', boom)

    body = client.post('/api/optimize', json={
        'strategy': 'double_ma', 'symbols': ['SA'],
        'start': '2020-01-01', 'end': '2022-01-01',
        'cash': 100_000.0, 'slippage': 0.0, 'n_trials': 2, 'n_folds': 2,
        'embargo': 10, 'holdout_frac': 0.2, 'lambda_std': 0.5,
        'min_trades_per_year': 4.0, 'dd_cap': 0.35, 'param_overrides': {},
        'seed': 1, 'probe_samples': 2,
    }).json()

    for _ in range(200):
        job = client.get(f"/api/jobs/{body['job_id']}").json()
        if job['status'] in ('done', 'error'):
            break
        time.sleep(0.1)
    assert job['status'] == 'error'

    run = client.get(f"/api/runs/{body['run_id']}").json()
    assert run['status'] == 'error', 'the run row was left mid-flight'
    assert 'no data for SA' in run['error']

    # And being terminal, it is now something the retention prune can reclaim.
    stale = store_module.list_runs(kind='optimize')
    assert all(r['status'] != 'running' for r in stale)


@pytest.mark.parametrize('path, model, extra', [
    ('/api/backtest', BacktestRequest, {}),
    ('/api/optimize', OptimizeRequest, {
        'n_trials': 2, 'n_folds': 2, 'embargo': 10, 'holdout_frac': 0.2,
        'lambda_std': 0.5, 'min_trades_per_year': 4.0, 'dd_cap': 0.35,
        'param_overrides': {}, 'seed': 1, 'probe_samples': 2,
    }),
])
def test_negative_slippage_is_refused(client, path, model, extra):
    """Slippage is a cost, so it can only move a fill against you. A negative
    one moves it *for* you and does not fail -- it just inflates the result and
    records it in history looking like any other run, which is worse than an
    error.

    The rejection goes through HTTP because a 422 is what the form has to
    render; the accepted values are checked on the model instead, so this test
    does not leave a real run on the shared job manager for whatever runs next.
    """
    base = {
        'strategy': 'double_ma', 'symbols': ['SA'],
        'start': '2024-01-01', 'end': '2024-06-01', 'cash': 200_000.0,
    }
    resp = client.post(path, json={**base, **extra, 'slippage': -1.0})
    assert resp.status_code == 422, resp.text
    assert 'greater than or equal to 0' in json.dumps(resp.json())

    # Zero is the documented default and stays valid, as does any real cost.
    assert model(**base, **extra, slippage=0.0).slippage == 0.0
    assert model(**base, **extra, slippage=1.5).slippage == 1.5
    assert model(**base, **extra).slippage == 0.0


@pytest.mark.parametrize('path', ['/api/backtest', '/api/optimize'])
def test_an_empty_symbol_list_is_a_422_not_a_500(client, path):
    """``require_products`` reports an unknown product as ``KeyError`` but an
    empty list as ``ValueError``, and both handlers used to catch only the
    first.

    An empty universe is exactly what the form posts when the symbol
    multi-select is submitted with nothing picked, so the likeliest bad
    submission on the page was the one that came back as an unhandled 500 --
    which the panel can only render as "Internal Server Error", with none of
    the field-level detail a 422 carries.

    Neither router reaches ``store.create_run`` before validating, so a
    rejected submission must also leave no run behind: a row stuck at
    ``running`` is never pruned (see the run-history tests below) and would
    outlive the mistake that made it.
    """
    before = len(store_module.list_runs(limit=1000))
    body = {'strategy': 'double_ma', 'start': '2024-01-01', 'end': '2024-06-01'}

    resp = client.post(path, json={**body, 'symbols': []})
    assert resp.status_code == 422, resp.text
    assert 'No products specified' in json.dumps(resp.json())

    # The KeyError path it shares the handler with is untouched.
    resp = client.post(path, json={**body, 'symbols': ['ZZZ']})
    assert resp.status_code == 422, resp.text
    assert 'ZZZ' in json.dumps(resp.json())

    assert len(store_module.list_runs(limit=1000)) == before


def test_a_run_that_starts_insolvent_is_reported_as_blown_up(client):
    """``cash <= 0`` is an account that is insolvent on its first bar: the
    engine flags the blow-up and stops, nothing is ever appended to
    ``equity_records``, and ``compute_metrics`` returns ``{}`` for an empty
    run -- dropping the ``blown_up`` it was handed. The job itself raises
    nothing, so this reached the panel as a green "done" over a grid of "n/a".
    The router carries the engine's own flag across that gap; the panel keys
    its red badge off it, so it has to survive both the job result and the
    stored run row.
    """
    body = client.post('/api/backtest', json={
        'strategy': 'double_ma', 'symbols': ['SA'],
        'start': '2024-01-01', 'end': '2024-06-01', 'cash': 0.0, 'slippage': 0.0,
    }).json()

    for _ in range(200):
        job = client.get(f"/api/jobs/{body['job_id']}").json()
        if job['status'] in ('done', 'error'):
            break
        time.sleep(0.1)
    assert job['status'] == 'done'
    assert job['result']['metrics']['blown_up'] is True

    run = client.get(f"/api/runs/{body['run_id']}").json()
    assert run['metrics']['blown_up'] is True

    # The history list is where the badge is drawn from, so the flag has to
    # survive the summary projection too, not just the detail endpoint.
    row = next(r for r in client.get('/api/runs?kind=backtest').json() if r['id'] == body['run_id'])
    assert row['status'] == 'done' and row['metrics']['blown_up'] is True

    # A solvent run over the same window must not be flagged.
    ok = client.post('/api/backtest', json={
        'strategy': 'double_ma', 'symbols': ['SA'],
        'start': '2024-01-01', 'end': '2024-06-01', 'cash': 200_000.0, 'slippage': 0.0,
    }).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{ok['job_id']}").json()
        if job['status'] in ('done', 'error'):
            break
        time.sleep(0.1)
    assert job['status'] == 'done'
    assert job['result']['metrics']['blown_up'] is False


def test_report_delete_removes_only_the_report_json(client, tmp_path, monkeypatch):
    """The router binds ``RESULTS_DIR`` under its own name at import time, so
    the directory has to be patched on the router module -- patching
    ``research.optimize.RESULTS_DIR`` (what the run_study test above does)
    would leave list/get/delete still pointed at the repo's real
    ``results/optuna/``.
    """
    import web.routers.optimize as optimize_router

    monkeypatch.setattr(optimize_router, 'OPTUNA_RESULTS_DIR', str(tmp_path))

    for name, evaluated in (('DoubleMa_20250101_best.json', False), ('DoubleMa_20250102_best.json', True)):
        (tmp_path / name).write_text(json.dumps({
            'strategy': 'DoubleMaStrategy', 'symbols': ['SA'], 'timestamp': name.split('_')[1],
            'best_value': 1.5, 'holdout_evaluated': evaluated,
        }), encoding='utf-8')
    # One db backs every run of a study name, so several reports can point at
    # it -- deleting one of them must not touch it.
    db = tmp_path / 'DoubleMaStrategy_SA.db'
    db.write_bytes(b'not really sqlite, but its survival is the point')

    listing = client.get('/api/optimize/reports').json()
    assert {r['name'] for r in listing} == {'DoubleMa_20250101_best.json', 'DoubleMa_20250102_best.json'}

    resp = client.delete('/api/optimize/reports/DoubleMa_20250101_best.json')
    assert resp.status_code == 200, resp.text
    assert resp.json() == {'deleted': 'DoubleMa_20250101_best.json'}

    assert not (tmp_path / 'DoubleMa_20250101_best.json').exists()
    assert (tmp_path / 'DoubleMa_20250102_best.json').exists()
    assert db.exists()

    listing = client.get('/api/optimize/reports').json()
    assert [r['name'] for r in listing] == ['DoubleMa_20250102_best.json']

    # Gone means gone, for both the second delete and any later read.
    assert client.delete('/api/optimize/reports/DoubleMa_20250101_best.json').status_code == 404
    assert client.get('/api/optimize/reports/DoubleMa_20250101_best.json').status_code == 404


def test_report_delete_answers_404_when_the_file_vanishes_mid_call(client, tmp_path, monkeypatch):
    """``_report_path`` checks existence and ``os.remove`` acts on the answer,
    so a second tab deleting the same report in between turned a lost race into
    a 500 traceback. It is the same "already gone" the endpoint answers 404 for
    when it loses the race by a wider margin."""
    import web.routers.optimize as optimize_router

    monkeypatch.setattr(optimize_router, 'OPTUNA_RESULTS_DIR', str(tmp_path))
    (tmp_path / 'DoubleMa_20250101_best.json').write_text('{}', encoding='utf-8')

    real_remove = os.remove

    def vanish(path):
        real_remove(path)                      # the other tab's delete
        real_remove(path)                      # ours, now losing the race

    monkeypatch.setattr(optimize_router.os, 'remove', vanish)

    resp = client.delete('/api/optimize/reports/DoubleMa_20250101_best.json')
    assert resp.status_code == 404, resp.text
    assert 'DoubleMa_20250101_best.json' in resp.json()['detail']


def test_report_delete_rejects_names_that_escape_the_results_dir(client, tmp_path, monkeypatch):
    """``_report_path`` is the only guard between a URL segment and
    ``os.remove``. httpx collapses ``..`` client-side, so the reachable
    attack is a name that stays one segment: an absolute-ish or
    separator-carrying name, or a non-report extension.
    """
    import web.routers.optimize as optimize_router

    monkeypatch.setattr(optimize_router, 'OPTUNA_RESULTS_DIR', str(tmp_path))

    victim = tmp_path / 'keepme.txt'
    victim.write_text('untouched', encoding='utf-8')

    for name in ('..%2F..%2Fdatafeed%2Fproducts.json', '..%5Cproducts.json', 'keepme.txt', 'DoubleMa_best'):
        resp = client.delete(f'/api/optimize/reports/{name}')
        assert resp.status_code in (400, 404), (name, resp.status_code)
    assert victim.read_text(encoding='utf-8') == 'untouched'


# ---------------------------------------------------------------------
# Path confinement
#
# The SPA catch-all's URL reaches the filesystem, and used to do so
# unfiltered. It is not reachable through the UI -- a browser collapses
# `..` before the request leaves -- which is exactly why the guard needs a
# test rather than manual checking.
# ---------------------------------------------------------------------

def _raw_get(path: str) -> tuple:
    """Drive the ASGI app with the scope uvicorn builds for ``path``.

    ``TestClient``/httpx normalises ``..`` out of a URL client-side, so it
    cannot express this attack at all -- but uvicorn only percent-decodes
    the request line, it does not collapse segments, so a raw client
    (``curl --path-as-is``, a proxy, anything non-browser) reaches the
    handler with the dot segments intact. Building the scope by hand is the
    only way to reproduce what the server actually sees.
    """
    import asyncio
    from urllib.parse import unquote

    # uvicorn percent-decodes the request line into scope['path'] and stops
    # there -- so `%2e%2e%2f` arrives at the handler as `../`, already
    # decoded but never collapsed. Decoding here is what makes the encoded
    # cases below test the same code path a real request would.
    scope = {
        'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
        'http_version': '1.1', 'method': 'GET', 'scheme': 'http',
        'path': unquote(path), 'raw_path': path.encode(), 'query_string': b'',
        'root_path': '', 'headers': [(b'host', b'127.0.0.1:8000')],
        'client': ('127.0.0.1', 1234), 'server': ('127.0.0.1', 8000),
    }
    captured = {'status': None, 'body': b''}

    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message):
        if message['type'] == 'http.response.start':
            captured['status'] = message['status']
        elif message['type'] == 'http.response.body':
            captured['body'] += message.get('body', b'')

    asyncio.run(app(scope, receive, send))
    return captured['status'], captured['body']


@pytest.mark.skipif(not os.path.isdir(STATIC_DIR), reason='no built frontend in web/static')
@pytest.mark.parametrize('path,marker', [
    ('/../../datafeed/products.json', b'"exchange"'),
    ('/../../results/webpanel.db', b'SQLite format'),
    ('/../../../../../../etc/passwd', b'root:'),
    ('/../../pyproject.toml', b'futurestoolkit'),
    ('/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd', b'root:'),
    ('/assets/../../../../../../../../etc/passwd', b'root:'),
])
def test_spa_route_refuses_to_serve_anything_outside_the_build_dir(path, marker):
    """Assert on the response *body*, not on a status code or a handler.

    ``/assets/...`` is claimed by the ``StaticFiles`` mount rather than the
    SPA route and is refused with a 404, while everything else falls through
    to the index.html shell with a 200 -- both are safe, and pinning either
    one would make this test about which handler matched instead of about
    whether a file escaped. What must hold in every case is that the target
    file's content never appears in the response.
    """
    status, body = _raw_get(path)
    assert status in (200, 404), status
    assert marker not in body, f'{path} leaked {body[:80]!r}'


@pytest.mark.skipif(not os.path.isdir(STATIC_DIR), reason='no built frontend in web/static')
def test_spa_route_still_serves_real_build_files():
    status, body = _raw_get('/favicon.svg')
    assert status == 200
    assert b'<svg' in body


# ---------------------------------------------------------------------
# Job log streaming
#
# Both defects below only appear on the long runs the SSE stream exists for
# (a 16-symbol data update, a few-hundred-trial study) and neither shows up
# as an error -- the log just stops, or arrives 15 seconds late. Short tests
# reproduce them by shrinking the buffer instead of by running long.
# ---------------------------------------------------------------------

def _collect_stream(manager, job_id, *, timeout=60.0):
    """Drain one job's SSE stream to completion, returning (log messages,
    state frame count). Runs its own loop, the way the endpoint's
    ``StreamingResponse`` would."""
    import asyncio

    async def drain():
        messages, states = [], 0
        async for chunk in manager.stream(job_id):
            for raw in chunk.split('\n'):
                if not raw.startswith('data: '):
                    continue
                payload = json.loads(raw[len('data: '):])
                if payload['type'] == 'log':
                    messages.append(payload['message'])
                else:
                    states += 1
        return messages, states

    return asyncio.run(asyncio.wait_for(drain(), timeout))


@pytest.fixture
def job_logger():
    """The engine loggers the manager listens on default to the root level;
    pin INFO so a test's log lines actually reach the handler."""
    log = logging.getLogger('futurestoolkit')
    previous = log.level
    log.setLevel(logging.INFO)
    yield log
    log.setLevel(previous)


def test_stream_keeps_delivering_after_the_log_buffer_wraps(job_logger, monkeypatch):
    """A job emitting more than JOB_LOG_BUFFER lines used to go silent.

    The stream tracked how many lines it had sent as a running count and
    sliced the buffer with it. That buffer is a ring: once it starts
    evicting, its length stops growing while the count does not, so the
    slice was empty from then on and every later line vanished with no
    error. Sequence numbers survive eviction; counts do not.
    """
    monkeypatch.setattr(jobs_module, 'JOB_LOG_BUFFER', 8)
    manager = JobManager(max_workers=1)
    n_lines = 120
    gate = threading.Event()

    def work(job):
        gate.wait(5)
        for i in range(n_lines):
            job_logger.info('line-%d', i)
            time.sleep(0.001)
        return 'ok'

    job = manager.submit('test', work)
    gate.set()
    messages, _ = _collect_stream(manager, job.id)

    delivered = [m for m in messages if m.startswith('line-')]
    notices = [m for m in messages if not m.startswith('line-')]
    indices = [int(m.split('-')[1]) for m in delivered]

    assert len(delivered) > jobs_module.JOB_LOG_BUFFER, (
        f'only {len(delivered)} lines got through a buffer of '
        f'{jobs_module.JOB_LOG_BUFFER} -- the stream stopped at the wrap'
    )
    assert indices == sorted(indices), 'lines arrived out of order'
    assert len(set(indices)) == len(indices), 'lines were sent twice'
    # Whatever did not arrive must have been announced, not silently dropped.
    announced = sum(int(n.split()[0].lstrip('[')) for n in notices)
    assert len(delivered) + announced == n_lines


def test_stream_announces_lines_that_scrolled_out_before_it_attached(job_logger, monkeypatch):
    """A client attaching to a job that already overflowed its buffer is told
    how much it missed, rather than being handed a log that looks whole."""
    monkeypatch.setattr(jobs_module, 'JOB_LOG_BUFFER', 5)
    manager = JobManager(max_workers=1)
    n_lines = 40

    def work(job):
        for i in range(n_lines):
            job_logger.info('line-%d', i)
        return 'ok'

    job = manager.submit('test', work)
    for _ in range(200):                      # let it finish before attaching
        if job.status in ('done', 'error'):
            break
        time.sleep(0.01)
    assert job.status == 'done'

    messages, _ = _collect_stream(manager, job.id)
    delivered = [m for m in messages if m.startswith('line-')]
    notices = [m for m in messages if not m.startswith('line-')]

    assert len(delivered) == jobs_module.JOB_LOG_BUFFER    # only the tail survived
    assert len(notices) == 1
    assert int(notices[0].split()[0].lstrip('[')) == n_lines - jobs_module.JOB_LOG_BUFFER


def test_stream_is_woken_by_a_log_line_not_by_its_own_timeout(job_logger):
    """`_notify` hands off from a worker thread to the loop serving the SSE
    response. Poking the asyncio.Queue directly did not wake that loop, so
    every update waited out the stream's 15s idle timeout -- "live" progress
    that was up to 15 seconds stale, and a cancel that took as long to show.
    """
    import asyncio

    manager = JobManager(max_workers=1)
    gate, release = threading.Event(), threading.Event()
    emitted = {}

    def work(job):
        gate.wait(5)
        emitted['at'] = time.monotonic()
        job_logger.info('THE-LINE')
        release.wait(30)          # hold the job open so the stream cannot end
        return 'ok'

    job = manager.submit('test', work)

    async def wait_for_line():
        async for chunk in manager.stream(job.id):
            for raw in chunk.split('\n'):
                if raw.startswith('data: '):
                    payload = json.loads(raw[len('data: '):])
                    if payload['type'] == 'log' and 'THE-LINE' in payload['message']:
                        return time.monotonic()
        return None

    async def main():
        task = asyncio.create_task(wait_for_line())
        await asyncio.sleep(0.3)              # let the stream reach its await
        gate.set()
        try:
            return await asyncio.wait_for(task, timeout=10.0)
        finally:
            release.set()

    received = asyncio.run(main())
    assert received is not None, 'log line never reached the stream'
    latency = received - emitted['at']
    # The bug parked this at ~15s (the idle timeout). Assert well under it
    # rather than near-zero, so a loaded CI box does not make this flaky.
    assert latency < 5.0, f'took {latency:.1f}s -- the stream waited out its timeout'


# ---------------------------------------------------------------------
# Guards around shared mutable state
# ---------------------------------------------------------------------

def test_unknown_api_path_is_a_json_404_not_the_spa_shell(client):
    """The SPA catch-all used to swallow these, so a misspelled endpoint came
    back 200 text/html and the frontend's fetch wrapper died on JSON.parse
    instead of surfacing the 404."""
    for method, path in [('get', '/api/nope'), ('get', '/api/runs/x/nope'),
                         ('post', '/api/nope'), ('delete', '/api/nope')]:
        resp = getattr(client, method)(path)
        assert resp.status_code == 404, f'{method.upper()} {path} -> {resp.status_code}'
        assert resp.headers['content-type'].startswith('application/json')
        assert 'No such endpoint' in resp.json()['detail']


def test_real_api_routes_still_win_over_the_catch_all(client):
    assert client.get('/api/health').json() == {'status': 'ok'}
    assert client.get('/api/products').status_code == 200


def test_foreground_work_is_visible_to_the_active_job_check():
    """Synchronous handlers have to register too, or the guard that holds off
    a registry edit mid-run cannot see them -- which is how a holdout
    evaluation, the one run whose numbers are spent exactly once, could be
    corrupted by a multiplier edit landing halfway through."""
    manager = JobManager(max_workers=1)
    assert not manager.any_active()
    with manager.foreground('holdout') as job:
        assert manager.any_active()
        assert job.status == 'running'
    assert not manager.any_active()
    assert manager.get(job.id).status == 'done'


def test_foreground_records_a_failure_and_still_re_raises():
    manager = JobManager(max_workers=1)
    with pytest.raises(ValueError, match='boom'):
        with manager.foreground('holdout') as job:
            raise ValueError('boom')
    assert not manager.any_active()
    assert manager.get(job.id).status == 'error'
    assert 'boom' in manager.get(job.id).error


def test_product_write_is_refused_while_foreground_work_runs(client, monkeypatch):
    """Same refusal the pooled-job test covers, for the inline path."""
    manager = jobs_module.manager
    with manager.foreground('holdout'):
        resp = client.put('/api/products/SA', json=dict(_VALID_PRODUCT))
        assert resp.status_code == 409
        assert 'holdout' in resp.json()['detail'] or 'job is running' in resp.json()['detail'].lower()


def test_second_update_of_the_same_symbol_is_refused(client, monkeypatch):
    """Two updates of one product write the same CSVs from independent
    fetches, so the loser silently overwrites the winner. Two pool workers
    means clicking Update twice is enough to reach it."""
    import web.routers.data as data_router

    class _NoopUpdate:
        """Stands in for the real downloader: this is about the claim/refuse
        logic, and the accepted request below must not go to the network."""
        stale_keys = ()

        def __init__(self, symbol):
            self.symbol = symbol

        def update(self, force=False, rebuild_only=False):
            return None

    monkeypatch.setattr(data_router, 'DataUpdate', _NoopUpdate)
    monkeypatch.setattr(data_router, '_inflight', {'SA'})

    resp = client.post('/api/data/update', json={'symbols': ['SA', 'CF']})
    assert resp.status_code == 409
    assert 'SA' in resp.json()['detail']

    # A disjoint set is unaffected.
    started = client.post('/api/data/update', json={'symbols': ['CF']})
    assert started.status_code == 200

    # Drain it before returning: a job left running is one `any_active()`
    # reports to every later test, which then sees its product writes refused.
    job_id = started.json()['job_id']
    for _ in range(300):
        if client.get(f'/api/jobs/{job_id}').json()['status'] in ('done', 'error'):
            break
        time.sleep(0.01)
    assert not data_router.job_manager.any_active()


def test_update_refuses_force_together_with_rebuild_only(client):
    """The CLI puts these two in a mutually exclusive group; the API used to
    accept both, and ``DataUpdate`` then skipped the sync and dropped
    ``force`` silently -- a request to re-download that never downloaded."""
    resp = client.post(
        '/api/data/update',
        json={'symbols': ['SA'], 'force': True, 'rebuild_only': True},
    )
    assert resp.status_code == 422
    assert 'mutually exclusive' in str(resp.json()['detail'])

    # Each mode on its own is still a valid request body.
    for body in ({'force': True}, {'rebuild_only': True}, {}):
        assert DataUpdateRequest(symbols=['SA'], **body).symbols == ['SA']


def test_run_history_write_does_not_deadlock_on_a_cold_connection():
    """`create_run` calls `_get_conn` while already holding the write lock, so
    building the connection under that same non-reentrant lock hangs the very
    first write of the process -- silently, with no error.

    Run on a worker thread with a bounded join: the regression is a hang, and
    a test that reproduced it by hanging would take CI down with it instead of
    reporting a failure.
    """
    store_module._conn = None
    result = {}

    def write():
        result['id'] = store_module.create_run(kind='backtest', strategy='X', symbols=['SA'])

    worker = threading.Thread(target=write, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), 'create_run deadlocked while building its connection'
    assert store_module.get_run(result['id'])['strategy'] == 'X'


def test_product_create_addresses_the_code_in_the_path(client):
    """POST used to take the code as a query parameter while PUT and DELETE
    took it in the path -- one identifier, two spellings depending on verb."""
    resp = client.post('/api/products/QQ', json=_VALID_PRODUCT)
    assert resp.status_code == 201
    assert resp.json()['code'] == 'QQ'
    # The old spelling is not a route any more, and lands on the /api 404.
    assert client.post('/api/products', json=_VALID_PRODUCT).status_code == 404


def test_progress_data_is_streamed_as_deltas_not_re_sent_each_frame():
    """A study appends one trial entry per callback and never trims, and the
    stream emits a state frame per callback -- echoing the whole list in each
    made the traffic quadratic in trial count. Deltas keep it linear; the
    entries still have to arrive exactly once each, in order.
    """
    manager = JobManager(max_workers=1)
    n_trials = 60
    gate = threading.Event()

    def work(job):
        gate.wait(5)
        for i in range(n_trials):
            job.progress_data.append({'trial': i, 'value': float(i)})
            manager.notify(job.id)
            time.sleep(0.001)
        return 'ok'

    job = manager.submit('optimize', work)
    gate.set()

    import asyncio

    async def drain():
        entries, state_frames_with_list = [], 0
        async for chunk in manager.stream(job.id):
            for raw in chunk.split('\n'):
                if not raw.startswith('data: '):
                    continue
                payload = json.loads(raw[len('data: '):])
                if payload['type'] == 'progress_data':
                    entries.extend(payload['entries'])
                elif payload['type'] == 'state' and 'progress_data' in payload:
                    state_frames_with_list += 1
        return entries, state_frames_with_list

    entries, state_frames_with_list = asyncio.run(asyncio.wait_for(drain(), 60))

    assert state_frames_with_list == 0, 'state frames still carry the whole list'
    assert [e['trial'] for e in entries] == list(range(n_trials))


def test_jobs_listing_omits_trial_telemetry_but_the_detail_view_keeps_it(client):
    manager = jobs_module.manager
    job = manager.submit('optimize', lambda j: j.progress_data.append({'trial': 0, 'value': 1.0}))
    for _ in range(200):
        if job.status in ('done', 'error'):
            break
        time.sleep(0.01)

    listed = next(j for j in client.get('/api/jobs').json() if j['id'] == job.id)
    assert 'progress_data' not in listed
    assert listed['status'] == 'done'

    detail = client.get(f'/api/jobs/{job.id}').json()
    assert detail['progress_data'] == [{'trial': 0, 'value': 1.0}]


# ---------------------------------------------------------------------
# A strategy name arriving over HTTP never reaches importlib
# ---------------------------------------------------------------------

def _import_probe(tmp_path, monkeypatch, name: str):
    """Put an importable module on ``sys.path`` that leaves a file behind if
    it is ever imported, and return that marker path.

    Asserting on the marker rather than on the response alone is the point:
    ``load_strategy``'s ``'module:Class'`` form imports first and checks the
    result is a ``Strategy`` second, so a 4xx is perfectly compatible with
    the module's top-level code having already run.
    """
    marker = tmp_path / f'{name}.imported'
    (tmp_path / f'{name}.py').write_text(
        'import pathlib\n'
        f'pathlib.Path({str(marker)!r}).write_text("imported")\n'
        'class NotAStrategy:\n'
        '    pass\n',
        encoding='utf-8',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return marker


def test_backtest_refuses_a_module_path_strategy_without_importing_it(client, tmp_path, monkeypatch):
    marker = _import_probe(tmp_path, monkeypatch, 'ftk_probe_backtest')
    resp = client.post('/api/backtest', json={
        'strategy': 'ftk_probe_backtest:NotAStrategy',
        'symbols': ['SA'], 'start': '2024-01-01', 'end': '2024-06-01',
    })
    assert resp.status_code == 422, resp.text
    assert not marker.exists(), 'the request body got to import a module'
    assert 'ftk_probe_backtest' not in sys.modules


def test_optimize_refuses_a_module_path_strategy_without_importing_it(client, tmp_path, monkeypatch):
    marker = _import_probe(tmp_path, monkeypatch, 'ftk_probe_optimize')
    resp = client.post('/api/optimize', json={
        'strategy': 'ftk_probe_optimize:NotAStrategy',
        'symbols': ['SA'], 'start': '2024-01-01', 'end': '2024-06-01',
    })
    assert resp.status_code == 422, resp.text
    assert not marker.exists(), 'the request body got to import a module'
    assert 'ftk_probe_optimize' not in sys.modules


def test_strategy_detail_refuses_a_module_path_key_without_importing_it(client, tmp_path, monkeypatch):
    marker = _import_probe(tmp_path, monkeypatch, 'ftk_probe_detail')
    resp = client.get('/api/strategies/ftk_probe_detail:NotAStrategy')
    assert resp.status_code == 404, resp.text
    assert not marker.exists(), 'the URL path got to import a module'
    assert 'ftk_probe_detail' not in sys.modules


def test_registry_only_loader_rejects_what_the_cli_loader_still_accepts():
    """The two loaders are deliberately not interchangeable.

    ``ft.py`` keeps the ``'module:Class'`` escape hatch -- an argv is already
    running as the user -- while every web entry point is restricted to names
    the registry actually discovered.
    """
    from strategies import load_registered_strategy, load_strategy

    assert load_strategy('strategies.double_ma:DoubleMaStrategy') is load_strategy('double_ma')
    assert load_registered_strategy('double_ma') is load_strategy('double_ma')
    with pytest.raises(KeyError):
        load_registered_strategy('strategies.double_ma:DoubleMaStrategy')


# ---------------------------------------------------------------------
# Run history is bounded, on disk as well as in the index
# ---------------------------------------------------------------------

def test_run_history_prunes_oldest_rows_and_their_artifact_files(monkeypatch):
    """Rows *and* files: an artifact left behind after its row is gone is
    unreachable from the panel, so it would leak megabytes per backtest with
    nothing able to find it again."""
    monkeypatch.setattr(store_module, 'RUN_RETENTION', 3)

    ids = []
    for i in range(6):
        run_id = store_module.create_run(kind='backtest', strategy=f'S{i}', symbols=['SA'])
        store_module.finish_run(run_id, status='done', metrics={'sharpe_ratio': i},
                                artifact={'equity_records': [{'date': '2024-01-01', 'equity': i}]})
        ids.append(run_id)

    kept = {r['id'] for r in store_module.list_runs(limit=100)}
    # Pruning happens after the INSERT and inside its transaction, so the new
    # row is already among the newest N: the table settles at exactly the cap.
    assert kept == set(ids[-3:]), kept
    for run_id in ids[:3]:
        assert store_module.get_run(run_id) is None
        assert not os.path.exists(os.path.join(store_module.WEB_RESULTS_DIR, f'{run_id}.json'))
    for run_id in ids[-3:]:
        assert os.path.exists(os.path.join(store_module.WEB_RESULTS_DIR, f'{run_id}.json'))


def test_run_history_prune_spares_a_run_still_in_flight(monkeypatch):
    """A long study outlived by a few hundred quick backtests must keep its
    row: `finish_run` updates by id, so a pruned row would turn that write
    into a silent no-op and lose the study's result."""
    monkeypatch.setattr(store_module, 'RUN_RETENTION', 2)

    long_run = store_module.create_run(kind='optimize', strategy='Study', symbols=['SA'])
    for i in range(5):
        quick = store_module.create_run(kind='backtest', strategy=f'S{i}', symbols=['SA'])
        store_module.finish_run(quick, status='done', metrics={'sharpe_ratio': i})

    assert store_module.get_run(long_run) is not None, 'a running study was pruned'
    store_module.finish_run(long_run, status='done', metrics={'sharpe_ratio': 9.0})
    assert store_module.get_run(long_run)['metrics']['sharpe_ratio'] == 9.0


def test_run_history_limit_cannot_be_widened_past_its_cap(client):
    """SQLite reads a negative ``LIMIT`` as *no* limit, so ``?limit=-1`` walked
    straight past the 100 the signature advertised and returned the whole
    table: every stored run's metrics and params, from one query string.
    """
    for i in range(3):
        run_id = store_module.create_run(kind='backtest', strategy=f'S{i}', symbols=['SA'])
        store_module.finish_run(run_id, status='done', metrics={'sharpe_ratio': i})

    for rejected in (-1, 0, 1001):
        resp = client.get(f'/api/runs?limit={rejected}')
        assert resp.status_code == 422, (rejected, resp.text)

    # The honest range still works, default included.
    assert len(client.get('/api/runs?limit=2').json()) == 2
    assert len(client.get('/api/runs').json()) == 3


# ---------------------------------------------------------------------
# Host header
# ---------------------------------------------------------------------

def test_a_foreign_host_header_is_refused():
    """The panel has no authentication and binds to loopback, which stops
    another *machine* but not another *page*: a site the user has open can
    resolve a name it controls to 127.0.0.1 and drive this API from inside
    their browser, where DELETE /api/products/{code}?purge_data=true deletes
    real files. Same-origin policy does not prevent the request, only the
    reading of its response, so the write lands either way.

    The Host header is the part such a request cannot forge -- it carries the
    attacker's own hostname -- so refusing anything the panel is not served
    under is what actually closes it.
    """
    attacker = TestClient(app, base_url='http://rebound.example.com')
    resp = attacker.get('/api/health')
    assert resp.status_code == 400, resp.text

    # A destructive verb is refused before it reaches any handler.
    resp = attacker.delete('/api/products/SA?purge_data=true')
    assert resp.status_code == 400, resp.text

    for allowed in ('http://localhost', 'http://127.0.0.1'):
        assert TestClient(app, base_url=allowed).get('/api/health').status_code == 200


def test_allowed_hosts_reads_the_environment(monkeypatch):
    """`ft.py web` widens the list through this variable when asked to bind
    somewhere other than loopback, where no default could guess the name the
    operator reaches the box by."""
    from web.config import ALLOWED_HOSTS_ENV, DEFAULT_ALLOWED_HOSTS, allowed_hosts

    monkeypatch.delenv(ALLOWED_HOSTS_ENV, raising=False)
    assert allowed_hosts() == list(DEFAULT_ALLOWED_HOSTS)

    monkeypatch.setenv(ALLOWED_HOSTS_ENV, 'panel.internal, 10.0.0.5 ,')
    assert allowed_hosts() == ['panel.internal', '10.0.0.5']

    monkeypatch.setenv(ALLOWED_HOSTS_ENV, '   ')
    assert allowed_hosts() == list(DEFAULT_ALLOWED_HOSTS)


def test_a_run_left_running_by_a_dead_server_is_adopted_on_reconnect(tmp_path, monkeypatch):
    """The job registry behind a ``running`` row is an in-memory dict, so a
    server that is killed mid-backtest leaves a row nothing can ever finish.

    That row was also the one status `_prune_locked` deliberately never
    deletes, so it was permanently ``running`` *and* permanently unprunable,
    while still occupying a slot in the newest-N window that keeps real
    history alive. Adopting it at connect time is safe precisely because of
    that in-memory registry: anything still ``running`` when a process first
    opens the index belongs to a process that is gone.
    """
    monkeypatch.setattr(store_module, 'DB_PATH', str(tmp_path / 'orphans.db'))
    monkeypatch.setattr(store_module, '_conn', None)

    stranded = store_module.create_run(kind='backtest', strategy='Killed', symbols=['SA'])
    finished = store_module.create_run(kind='backtest', strategy='Fine', symbols=['SA'])
    store_module.finish_run(finished, status='done', metrics={'sharpe_ratio': 1.0})
    assert store_module.get_run(stranded)['status'] == 'running'

    # Drop the connection the way a restart does, then reopen the same file.
    store_module._conn = None
    row = store_module.get_run(stranded)
    assert row['status'] == 'interrupted'
    assert 'never written' in row['error']

    # A row that already had a terminal status is left exactly as it was.
    assert store_module.get_run(finished)['status'] == 'done'
    assert store_module.get_run(finished)['error'] is None

    # And the adopted row is now prunable, which is the leak this closes.
    monkeypatch.setattr(store_module, 'RUN_RETENTION', 1)
    newest = store_module.create_run(kind='backtest', strategy='New', symbols=['SA'])
    store_module.finish_run(newest, status='done', metrics={'sharpe_ratio': 2.0})
    assert store_module.get_run(stranded) is None


def test_data_update_refuses_an_empty_selection_but_omitting_it_means_all(client, monkeypatch):
    """``[]`` and "the field is absent" used to be the same request.

    The panel's picker always posts an explicit array, so submitting it with
    nothing selected fell through to `list_products()` and started a download
    of the entire catalogue -- close to an hour on a cold cache, off a click
    that asked for no products at all. Omitting the field is still how a
    caller asks for everything, because that is a request someone can only
    make on purpose.
    """
    import web.routers.data as data_router

    class _NoopUpdate:
        stale_keys = ()

        def __init__(self, symbol):
            self.symbol = symbol

        def update(self, force=False, rebuild_only=False):
            return None

    monkeypatch.setattr(data_router, 'DataUpdate', _NoopUpdate)

    resp = client.post('/api/data/update', json={'symbols': []})
    assert resp.status_code == 422, resp.text
    assert 'No products specified' in json.dumps(resp.json())

    resp = client.post('/api/data/update', json={'symbols': ['ZZZ']})
    assert resp.status_code == 422, resp.text

    started = client.post('/api/data/update', json={})
    assert started.status_code == 200, started.text
    job_id = started.json()['job_id']
    for _ in range(500):
        if client.get(f'/api/jobs/{job_id}').json()['status'] in ('done', 'error'):
            break
        time.sleep(0.01)
    result = client.get(f'/api/jobs/{job_id}').json()
    assert result['status'] == 'done', result
    assert result['result']['symbols'] == products_module.list_products()

    # Same rule on the model itself, so the CLI-shaped call reads the same way.
    assert DataUpdateRequest().symbols is None
    assert DataUpdateRequest(symbols=['SA']).symbols == ['SA']

    assert not data_router.job_manager.any_active()
