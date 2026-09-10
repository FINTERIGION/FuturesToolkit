"""Run a single backtest and persist it into run history.

Thin wrapper over ``core.backtest.run_single_backtest`` -- no plotting, no
file writes beyond the run-history artifact -- the same function the
``ft.py backtest`` CLI already uses.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.backtest import run_single_backtest
from datafeed.products import require_products
from strategies import load_registered_strategy

from web import store
from web.jobs import manager as job_manager
from web.marketcache import cache as market_cache
from web.schemas import BacktestRequest
from web.serialize import annotate_inf_metrics

router = APIRouter(prefix='/api', tags=['backtest'])
logger = logging.getLogger('futurestoolkit.web')


def _build_artifact(market, symbols, result: dict) -> dict:
    """Everything a chart or a trade table needs, keyed for lazy per-symbol
    fetch (``/api/runs/{id}/price/{symbol}``) rather than shipped inline."""
    price = {}
    for sym in symbols:
        panel = market.products.get(sym)
        if panel is None:
            continue
        w = panel.weighted
        price[sym] = {
            'dates': [str(d) for d in market.dates],
            'open': w['open'].tolist(), 'high': w['high'].tolist(),
            'low': w['low'].tolist(), 'close': w['close'].tolist(),
            'volume': w['volume'].tolist(), 'oi': w['oi'].tolist(),
        }
    signals_by_symbol: dict = {}
    for row in result['signal_log']:
        signals_by_symbol.setdefault(row['symbol'], []).append(row)

    return {
        'equity_records': result['equity_records'],
        'trade_logs': result['trade_logs'],
        'signal_log_by_symbol': signals_by_symbol,
        'price': price,
        'deferred': result['deferred'],
    }


@router.post('/backtest')
def start_backtest(body: BacktestRequest):
    # ``ValueError`` alongside ``KeyError``: an unknown product raises the
    # latter, but an *empty* symbol list raises the former
    # (``require_products``), and that is what the form sends whenever the
    # universe multi-select is submitted with nothing picked. Catching only
    # ``KeyError`` turned the single most likely bad submission into a 500.
    try:
        symbols = require_products(body.symbols)
        strategy_cls = load_registered_strategy(body.strategy)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    run_id = store.create_run(
        kind='backtest', strategy=strategy_cls.__name__, symbols=symbols,
        start=body.start, end=body.end, cash=body.cash, slippage=body.slippage,
        params=body.params,
    )

    def run(job):
        job.message = 'Loading market data'
        market = market_cache.get(symbols, body.start, body.end)
        job.progress = 0.3
        job.message = f'Running {strategy_cls.__name__}'
        outcome = run_single_backtest(
            market, strategy_cls, body.params, body.cash, body.slippage,
        )
        job.progress = 0.9
        result, metrics = outcome['result'], outcome['metrics']
        # `compute_metrics` returns `{}` outright for a run with no recorded
        # bars, and that is exactly what an account which starts insolvent
        # (cash <= 0) produces: the engine flags the blow-up on bar 0 and
        # stops, so nothing is ever appended to `equity_records` and the
        # `blown_up` it was handed never makes it into the dict. The run then
        # reached the panel as a clean, empty success -- a green "done" over a
        # grid of "n/a". Carry the engine's own flag through so the panel can
        # say what actually happened. A no-op for a normal run, where
        # `metrics['blown_up']` already holds this same value.
        metrics = {**metrics, 'blown_up': bool(result['blown_up'])}
        artifact = _build_artifact(market, symbols, result)
        store.finish_run(run_id, status='done', metrics=metrics, artifact=artifact)
        job.progress = 1.0
        return {'run_id': run_id, 'metrics': annotate_inf_metrics(metrics)}

    def run_wrapped(job):
        try:
            return run(job)
        except Exception as exc:
            store.finish_run(run_id, status='error', error=str(exc))
            raise

    job = job_manager.submit('backtest', run_wrapped)
    return {'job_id': job.id, 'run_id': run_id}
