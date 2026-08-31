"""Optuna-driven parameter optimization: strategy-agnostic anchored
walk-forward search with holdout isolation and post-hoc overfitting
diagnostics.

``run_study`` is the only entry point research_runner.py needs. It:

1. Loads market data once.
2. Resolves the search space (declared / inferred / CLI-overridden).
3. Probes a handful of random configurations to size a warmup pad long
   enough for indicators across the whole space, then lays out anchored
   walk-forward folds plus a locked trailing holdout window.
4. Runs an Optuna study whose objective is the cross-fold-penalized mean
   out-of-sample score (``research.objective.fold_objective``); the
   holdout window is never touched here.
5. Computes IS/OOS decay, PBO (CSCV), Deflated Sharpe, and a parameter
   plateau check for the best trial, and writes everything to a JSON
   report next to the study's SQLite database.
"""

from __future__ import annotations

import datetime
import json
import logging
import os

import numpy as np
import optuna
from optuna.trial import TrialState

from core.params import Categorical, Float, Int
from strategies import load_strategy, name_for
from research.objective import fold_objective, score, window_years
from research.overfit import deflated_sharpe_ratio, is_oos_decay, pbo_cscv, plateau_check
from research.runner_api import load_market, run_window
from research.space import check_constraints, resolve_space, suggest
from research.splits import Window, anchored_walk_forward
from research.warmup import probe_warmup

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'optuna')

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _spec_to_json(spec) -> dict:
    if isinstance(spec, Int):
        return {'kind': 'int', 'low': spec.low, 'high': spec.high, 'step': spec.step, 'log': spec.log}
    if isinstance(spec, Float):
        return {'kind': 'float', 'low': spec.low, 'high': spec.high, 'step': spec.step, 'log': spec.log}
    if isinstance(spec, Categorical):
        return {'kind': 'categorical', 'choices': list(spec.choices)}
    raise TypeError(f'Unknown space spec: {spec!r}')


def _probe_reserve_bars(market, strategy_cls, space, *, n_samples: int, margin: float, seed: int) -> int:
    """Sample ``n_samples`` random configurations from ``space`` and take
    the largest observed warmup, inflated by ``margin``, as the number of
    leading bars every fold reserves for indicator history. Strategy-
    agnostic: this is just as valid for a 2-parameter crossover as for a
    10-parameter factor model.
    """
    sampler = optuna.samplers.RandomSampler(seed=seed)
    probe_study = optuna.create_study(direction='maximize', sampler=sampler)
    defaults = dict(getattr(strategy_cls, 'params', {}) or {})

    warmups = [0]
    for _ in range(n_samples):
        trial = probe_study.ask()
        params = {**defaults, **suggest(trial, space)}
        if check_constraints(strategy_cls, params):
            warmups.append(probe_warmup(market, strategy_cls, params))
        probe_study.tell(trial, 0.0)

    return int(max(warmups) * margin) + 1


def run_study(
    *,
    strategy_cls: type,
    symbols,
    start: str = None,
    end: str = None,
    cash: float = 100_000.0,
    slippage: float = 0.0,
    n_trials: int = 200,
    n_folds: int = 4,
    embargo: int = 10,
    holdout_frac: float = 0.20,
    lambda_std: float = 0.5,
    min_trades_per_year: float = 4.0,
    dd_cap: float = 0.35,
    param_overrides: dict = None,
    seed: int = 42,
    probe_samples: int = 20,
    study_name: str = None,
    results_dir: str = None,
    update_data: bool = False,
    market=None,
) -> dict:
    """``market``: pass a pre-built ``MarketData`` (e.g. in tests, with
    synthetic data) to skip loading from ``DataManager`` entirely; ``start``/
    ``end`` are then unused except as report metadata. Otherwise ``market``
    is loaded from ``symbols``/``start``/``end`` as usual.
    """
    if market is None:
        if start is None or end is None:
            raise ValueError('Provide either `market`, or both `start` and `end` to load one.')
        market = load_market(symbols, start, end, update=update_data)
    space = resolve_space(strategy_cls, param_overrides)
    defaults = dict(getattr(strategy_cls, 'params', {}) or {})

    reserve_bars = _probe_reserve_bars(
        market, strategy_cls, space, n_samples=probe_samples, margin=1.2, seed=seed,
    )
    folds, holdout = anchored_walk_forward(
        market.n_bars, reserve_bars=reserve_bars, n_folds=n_folds,
        embargo=embargo, holdout_frac=holdout_frac,
    )
    # `holdout` is used below only to record its bar range in the report --
    # it is never passed to run_window. The `holdout` subcommand is the only
    # code path allowed to actually backtest it.
    full_window = Window('full', reserve_bars, holdout.start)
    pad = reserve_bars

    logger.info(
        "%s: %d bars total, reserve=%d, %d folds, embargo=%d, holdout=[%d,%d) (%d bars)",
        strategy_cls.__name__, market.n_bars, reserve_bars, n_folds, embargo,
        holdout.start, holdout.end, holdout.n_bars,
    )

    trial_returns: dict = {}

    def objective(trial: optuna.Trial) -> float:
        params = {**defaults, **suggest(trial, space)}
        if not check_constraints(strategy_cls, params):
            raise optuna.TrialPruned()

        train_scores, valid_scores, fold_metrics = [], [], []
        for train_w, valid_w in folds:
            train_out = run_window(market, strategy_cls, params, train_w, cash=cash, slippage=slippage, pad=pad)
            valid_out = run_window(market, strategy_cls, params, valid_w, cash=cash, slippage=slippage, pad=pad)
            train_scores.append(score(
                train_out['metrics'], window_years=window_years(train_w.n_bars),
                min_trades_per_year=min_trades_per_year, dd_cap=dd_cap,
            ))
            valid_scores.append(score(
                valid_out['metrics'], window_years=window_years(valid_w.n_bars),
                min_trades_per_year=min_trades_per_year, dd_cap=dd_cap,
            ))
            fold_metrics.append({'train': train_out['metrics'], 'valid': valid_out['metrics']})

        full_out = run_window(market, strategy_cls, params, full_window, cash=cash, slippage=slippage, pad=pad)
        returns = np.array(
            [r['daily_return'] for r in full_out['result']['equity_records']], dtype='float64',
        )
        if len(returns) < full_window.n_bars:
            returns = np.concatenate([np.zeros(full_window.n_bars - len(returns)), returns])
        trial_returns[trial.number] = returns

        trial.set_user_attr('params', params)
        trial.set_user_attr('train_scores', train_scores)
        trial.set_user_attr('valid_scores', valid_scores)
        trial.set_user_attr('fold_metrics', fold_metrics)

        return fold_objective(valid_scores, lambda_std=lambda_std)

    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    name = study_name or f"{strategy_cls.__name__}_{'-'.join(sorted(symbols))}"
    storage = f"sqlite:///{os.path.join(out_dir, name)}.db"

    study = optuna.create_study(
        study_name=name, storage=storage, load_if_exists=True,
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.number in trial_returns]
    if not completed:
        raise RuntimeError('No trial completed successfully -- check constraints/search space.')

    # Pick the best among *this run's* trials rather than `study.best_trial`.
    # With `load_if_exists=True` a resumed study also holds trials from earlier
    # runs, whose per-trial return series were never collected into
    # `trial_returns` -- and the PBO/DSR diagnostics below are computed against
    # that matrix, so an all-time winner from a previous run has nothing to be
    # diagnosed against.
    best_idx, best_trial = max(enumerate(completed), key=lambda pair: pair[1].value)
    best_params = best_trial.user_attrs['params']

    if study.best_trial.number != best_trial.number:
        logger.warning(
            "Study %r was resumed: its all-time best is trial #%d (value=%.4f) from an "
            "earlier run, but this run only re-ran %d trial(s), so the report below "
            "describes trial #%d (value=%.4f) -- the best of *this* run. Pass a fresh "
            "--study-name to search from scratch, or delete %s to reset.",
            name, study.best_trial.number, study.best_trial.value, len(completed),
            best_trial.number, best_trial.value, storage,
        )

    returns_matrix = np.stack([trial_returns[t.number] for t in completed], axis=0)

    def _evaluate(candidate_params: dict) -> float:
        valid_scores = []
        for train_w, valid_w in folds:
            valid_out = run_window(market, strategy_cls, candidate_params, valid_w, cash=cash, slippage=slippage, pad=pad)
            valid_scores.append(score(
                valid_out['metrics'], window_years=window_years(valid_w.n_bars),
                min_trades_per_year=min_trades_per_year, dd_cap=dd_cap,
            ))
        return fold_objective(valid_scores, lambda_std=lambda_std)

    diagnostics = {
        'is_oos_decay': is_oos_decay(best_trial.user_attrs['train_scores'], best_trial.user_attrs['valid_scores']),
        'pbo': pbo_cscv(returns_matrix, n_blocks=16),
        'dsr': deflated_sharpe_ratio(returns_matrix, best_idx),
        'plateau': plateau_check(strategy_cls, space, best_params, _evaluate),
    }

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'strategy': strategy_cls.__name__,
        'strategy_key': name_for(strategy_cls),
        'timestamp': ts,
        'study_name': name,
        'storage': storage,
        'symbols': sorted(symbols),
        'start': start,
        'end': end,
        'n_bars': market.n_bars,
        'n_trials_requested': n_trials,
        'n_trials_completed': len(completed),
        'n_folds': n_folds,
        'embargo': embargo,
        'reserve_bars': reserve_bars,
        'holdout_frac': holdout_frac,
        'holdout_window': {'start': holdout.start, 'end': holdout.end},
        'space': {k: _spec_to_json(v) for k, v in space.items()},
        'fixed_defaults': {k: v for k, v in defaults.items() if k not in space},
        'best_trial_number': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_params,
        'fold_train_scores': best_trial.user_attrs['train_scores'],
        'fold_valid_scores': best_trial.user_attrs['valid_scores'],
        'fold_metrics': best_trial.user_attrs['fold_metrics'],
        'diagnostics': diagnostics,
        'holdout_evaluated': False,
        'holdout_runs': 0,
        'cash': cash,
        'slippage': slippage,
    }

    path = os.path.join(out_dir, f'{strategy_cls.__name__}_{ts}_best.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info('Best trial #%d: objective=%.4f  params=%s', best_trial.number, best_trial.value, best_params)
    logger.info(
        'Diagnostics: IS/OOS ratio=%.2f  PBO=%.2f  DSR=%.2f  plateau_spikes=%s',
        diagnostics['is_oos_decay']['ratio'], diagnostics['pbo']['pbo'], diagnostics['dsr']['dsr'],
        [k for k, v in diagnostics['plateau']['dimensions'].items() if v['flags_spike']],
    )
    logger.info('Report written: %s', path)

    return {'path': path, 'report': report}


def evaluate_holdout(report_path: str, *, force: bool = False) -> dict:
    """Run ``best_params`` from an ``optimize`` report on its own locked
    holdout window, exactly once, and write the result back into the same
    JSON file.

    This is the second half of the leak-proof protocol: ``run_study`` above
    is structurally unable to reach the holdout window at all, so this is
    the *only* code path that ever backtests it. Calling it more than once
    for the same report -- without ``force`` -- is refused, because looking
    at holdout performance and then continuing to tune params on it turns
    holdout into a second validation set, silently invalidating the whole
    point of holding it out.
    """
    with open(report_path, encoding='utf-8') as f:
        report = json.load(f)

    if report.get('holdout_evaluated') and not force:
        raise RuntimeError(
            f"Holdout for {report_path!r} was already evaluated "
            f"{report.get('holdout_runs', 0)} time(s) on "
            f"{report.get('holdout_last_run')}. Re-checking holdout and then "
            f"continuing to tune turns it into a second validation set -- pass "
            f"force=True only if you understand and accept that, and treat any "
            f"result after the first as informational, not decision-making."
        )

    strategy_cls = load_strategy(report['strategy_key'])
    market = load_market(report['symbols'], report['start'], report['end'])
    holdout = Window('holdout', report['holdout_window']['start'], report['holdout_window']['end'])
    pad = report['reserve_bars']

    out = run_window(
        market, strategy_cls, report['best_params'], holdout,
        cash=report['cash'], slippage=report['slippage'], pad=pad,
    )
    holdout_score = score(
        out['metrics'], window_years=window_years(holdout.n_bars),
        min_trades_per_year=4.0, dd_cap=0.35,
    )

    report['holdout_evaluated'] = True
    report['holdout_runs'] = report.get('holdout_runs', 0) + 1
    report['holdout_last_run'] = datetime.datetime.now().isoformat(timespec='seconds')
    report['holdout_metrics'] = out['metrics']
    report['holdout_score'] = holdout_score

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    if report['holdout_runs'] > 1:
        logger.warning(
            '*** HOLDOUT EVALUATED %d TIMES for %s -- treat this run as informational '
            'only; any decision made after seeing a prior holdout result is no longer '
            'a clean out-of-sample check. ***', report['holdout_runs'], report_path,
        )

    logger.info(
        'Holdout [%d, %d): sharpe=%.4f  n_trades=%d  max_dd=%.2f%%  score=%.4f',
        holdout.start, holdout.end,
        out['metrics'].get('sharpe_ratio', 0.0), out['metrics'].get('n_trades', 0),
        out['metrics'].get('max_drawdown', 0.0), holdout_score,
    )

    return {'report': report, 'metrics': out['metrics'], 'score': holdout_score}
