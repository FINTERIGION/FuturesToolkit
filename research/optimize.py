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

from strategies import load_strategy, name_for
from research.objective import (
    DEFAULT_SPARSE_PENALTY, fold_objective, score, window_years,
)
from research.overfit import deflated_sharpe_ratio, is_oos_decay, pbo_cscv, plateau_check
from research.runner_api import load_market, run_window
from research.space import check_constraints, resolve_space, spec_to_json, suggest
from research.splits import Window, anchored_walk_forward
from research.warmup import probe_warmup

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'optuna')

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    sparse_penalty: float = DEFAULT_SPARSE_PENALTY,
    param_overrides: dict = None,
    seed: int = 42,
    probe_samples: int = 20,
    study_name: str = None,
    results_dir: str = None,
    update_data: bool = False,
    market=None,
    callbacks: list = None,
) -> dict:
    """``market``: pass a pre-built ``MarketData`` (e.g. in tests, with
    synthetic data) to skip loading from ``DataManager`` entirely; ``start``/
    ``end`` are then unused except as report metadata. Otherwise ``market``
    is loaded from ``symbols``/``start``/``end`` as usual.

    ``callbacks``: forwarded verbatim to ``optuna.study.Study.optimize`` --
    each is called as ``callback(study, trial)`` after every completed trial.
    The web panel uses this to stream progress and to stop the study early
    (``study.stop()``) on a user cancel; the CLI passes nothing, so a plain
    ``run_study`` call behaves exactly as before.
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
                sparse_penalty=sparse_penalty,
            ))
            valid_scores.append(score(
                valid_out['metrics'], window_years=window_years(valid_w.n_bars),
                min_trades_per_year=min_trades_per_year, dd_cap=dd_cap,
                sparse_penalty=sparse_penalty,
            ))
            fold_metrics.append({'train': train_out['metrics'], 'valid': valid_out['metrics']})

        full_out = run_window(market, strategy_cls, params, full_window, cash=cash, slippage=slippage, pad=pad)
        # Place the curve at the bar it actually starts on, and leave whatever
        # it does not cover at zero. A run can come up short at either end and
        # for opposite reasons: short at the *head* means the indicators needed
        # more pad than they got, short at the *tail* means the account blew up
        # and the run stopped there. Zero-filling the head unconditionally --
        # which is what this did -- shifted every blown-up trial's return
        # series forward in time, and `pbo_cscv` compares trials block by block
        # along exactly that axis, so a shifted trial was scored against the
        # wrong period in every single split. `effective_start` is absolute, in
        # `market`'s own bar numbering, so the offset needs no guessing.
        records = full_out['result']['equity_records']
        returns = np.zeros(full_window.n_bars, dtype='float64')
        if records:
            offset = max(0, full_out['effective_start'] - full_window.start)
            values = np.array([r['daily_return'] for r in records], dtype='float64')
            values = values[: full_window.n_bars - offset]
            returns[offset: offset + len(values)] = values
        trial_returns[trial.number] = returns

        trial.set_user_attr('params', params)
        trial.set_user_attr('train_scores', train_scores)
        trial.set_user_attr('valid_scores', valid_scores)
        trial.set_user_attr('fold_metrics', fold_metrics)
        # A trial that ran out of capital has no out-of-sample stretch at all:
        # the zeros standing in for the bars it never traded read to
        # `_period_sharpe` as a calm patch rather than a dead account, which
        # flatters its rank in every block after it died. Recorded so the
        # report can say how much of the PBO/DSR sample is in that state.
        trial.set_user_attr('blown_up', bool(full_out['result']['blown_up']))

        return fold_objective(valid_scores, lambda_std=lambda_std)

    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    name = study_name or f"{strategy_cls.__name__}_{'-'.join(sorted(symbols))}"
    storage = f"sqlite:///{os.path.join(out_dir, name)}.db"

    study = optuna.create_study(
        study_name=name, storage=storage, load_if_exists=True,
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

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

    # The search's own answer can be "nothing here trades", and that has to
    # read as a finding rather than as a result. Without this the degenerate
    # configuration comes back as `best_params` with diagnostics computed on a
    # flat equity curve, which do not look bad -- they look unremarkable.
    best_valid_trades = sum(
        f['valid'].get('n_trades', 0) for f in best_trial.user_attrs['fold_metrics']
    )
    if best_valid_trades == 0:
        logger.warning(
            "Best trial #%d never opened a position in any validation fold. That is "
            "not a tuned parameter set -- it is the search reporting that nothing in "
            "this space traded. PBO and DSR below are computed on a flat curve and "
            "will read as unremarkable rather than as bad. Widen the space, lengthen "
            "the window, or read this as 'no edge found'.", best_trial.number,
        )

    n_blown_up = sum(1 for t in completed if t.user_attrs.get('blown_up'))
    if n_blown_up:
        logger.warning(
            "%d of %d diagnosed trial(s) blew up mid-window. The bars they never "
            "traded stand in as zero returns, which reads as a calm stretch rather "
            "than a dead account, so PBO and DSR understate how bad those trials "
            "were. Treat both as optimistic while this count is high.",
            n_blown_up, len(completed),
        )

    returns_matrix = np.stack([trial_returns[t.number] for t in completed], axis=0)

    def _evaluate(candidate_params: dict) -> float:
        valid_scores = []
        for train_w, valid_w in folds:
            valid_out = run_window(market, strategy_cls, candidate_params, valid_w, cash=cash, slippage=slippage, pad=pad)
            valid_scores.append(score(
                valid_out['metrics'], window_years=window_years(valid_w.n_bars),
                min_trades_per_year=min_trades_per_year, dd_cap=dd_cap,
                sparse_penalty=sparse_penalty,
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
        # Stored so `evaluate_holdout` scores on the same scale the search
        # ranked on. It used to hardcode the defaults, which silently put the
        # holdout on a different objective from every fold whenever these were
        # tuned from the CLI.
        'min_trades_per_year': min_trades_per_year,
        'dd_cap': dd_cap,
        'sparse_penalty': sparse_penalty,
        'n_trials_blown_up': n_blown_up,
        'best_valid_trades': best_valid_trades,
        'reserve_bars': reserve_bars,
        'holdout_frac': holdout_frac,
        'holdout_window': {'start': holdout.start, 'end': holdout.end},
        'space': {k: spec_to_json(v) for k, v in space.items()},
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
        min_trades_per_year=report.get('min_trades_per_year', 4.0),
        dd_cap=report.get('dd_cap', 0.35),
        sparse_penalty=report.get('sparse_penalty', DEFAULT_SPARSE_PENALTY),
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
