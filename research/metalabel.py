"""Meta-labeling: fit a classifier that predicts, for each entry signal a
rule-based strategy fires, whether that trade will be profitable -- then
gate live entries on ``P(win) >= threshold`` via
``research.gating.MetaFilteredStrategy``.

Strategy-agnostic end to end: ``fit_meta_model`` only needs a ``Strategy``
subclass and its params; it discovers entry events via
``research.gating.make_recording_bar_context`` (no strategy-specific
parsing), features via ``research.features.build_feature_matrix`` (which
itself harvests whatever indicators the strategy registered), and labels
directly from the ledger's ``net_pnl``.

Anti-leakage discipline:

- Labels are ``net_pnl > 0``, sample weight ``|net_pnl|`` (so a low-win-rate,
  big-winner strategy doesn't get its tail chopped off just to raise
  accuracy) -- see the module docstring risk notes in the plan.
- Every event's ``[open_bar, close_bar]`` interval is known (``core/ledger``
  now carries both), so ``purge_events`` can drop any training event whose
  *outcome* was determined by price action inside -- or within ``embargo``
  bars of -- the validation window, the same purge discipline used in
  quantitative finance CV (Lopez de Prado, *Advances in Financial Machine
  Learning*, ch. 7).
- The threshold is chosen from out-of-fold predictions only, by expected
  value of the trades it would keep -- never by accuracy/AUC, which is
  exactly the objective that would chop off big winners.
- The final deployed model is refit on every usable (non-holdout) event
  only after fold-based selection is done -- there is no more "future" left
  relative to that data to leak into the fit itself.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from core.market import MarketData
from core.metrics import compute_metrics
from strategies import load_strategy
from research.features import build_feature_matrix
from research.gating import MetaFilteredStrategy, make_recording_bar_context
from research.runner_api import run_window, slice_start
from research.splits import Window, anchored_walk_forward

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    'logit': lambda seed=0: LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed),
    'rf': lambda seed=0: RandomForestClassifier(
        max_depth=3, n_estimators=200, class_weight='balanced', random_state=seed,
    ),
}


# ---------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------

def extract_events(market: MarketData, strategy_cls: type, params: dict, window: Window, *, cash, slippage=0.0, pad=0) -> dict:
    """Run ``strategy_cls(**params)`` once over ``window`` (isolated, via
    ``run_window``) with a recording ``BarContext`` wrapper, and match every
    flat->open or reversal intent to the ledger trade it produced (matched
    in order per symbol -- each such intent starts exactly one new ledger
    row, by construction of ``core.ledger``).

    Returns ``{'events': [...], 'run': run_window(...)'s full output}``.
    Each event: ``symbol, signal_bar, direction (+1/-1), open_bar,
    close_bar, net_pnl, label, weight, forced, open_at_end``.
    """
    log: list = []
    recording_cls = make_recording_bar_context(log)
    out = run_window(
        market, strategy_cls, params, window, cash=cash, slippage=slippage, pad=pad,
        bar_context_cls=recording_cls,
    )
    # `log` was built by a BarContext running against a slice starting at
    # bar `window.start - out['lo']`, so every `rec['bar']` is slice-local --
    # normalize to the same absolute numbering `run_window` already applied
    # to trade_logs' open_bar/close_bar, or matching against them below
    # would silently misalign by `lo` bars whenever pad != window.start.
    if out['lo']:
        for rec in log:
            rec['bar'] += out['lo']

    def is_entry(rec):
        start, target = rec['start'], rec['target']
        return target != 0 and (start == 0 or (start > 0) != (target > 0))

    entries_by_symbol: dict = {}
    for rec in log:
        if is_entry(rec):
            entries_by_symbol.setdefault(rec['symbol'], []).append(rec)

    trades_by_symbol: dict = {}
    for t in out['result']['trade_logs']:
        trades_by_symbol.setdefault(t['symbol'], []).append(t)

    events = []
    for sym, entries in entries_by_symbol.items():
        entries = sorted(entries, key=lambda r: r['bar'])
        trades = sorted(trades_by_symbol.get(sym, []), key=lambda t: t['open_bar'])
        if len(entries) != len(trades):
            logger.warning(
                "%s/%s: %d entry intents but %d ledger trades over %s -- matching "
                "the first %d pairs only.",
                strategy_cls.__name__, sym, len(entries), len(trades), window.name,
                min(len(entries), len(trades)),
            )
        for rec, trade in zip(entries, trades):
            net_pnl = trade['net_pnl']
            events.append({
                'symbol': sym,
                'signal_bar': rec['bar'],
                'direction': 1 if rec['target'] > 0 else -1,
                'open_bar': trade['open_bar'],
                'close_bar': trade['close_bar'],
                'net_pnl': net_pnl,
                'label': net_pnl > 0,
                'weight': abs(net_pnl),
                'forced': bool(trade['forced']),
                'open_at_end': bool(trade['open_at_end']),
            })

    events.sort(key=lambda e: e['signal_bar'])
    return {'events': events, 'run': out}


def position_flags(n_bars: int, run: dict, symbols) -> dict:
    """``{symbol: np.ndarray[n_bars]}`` of 0/1, built from one isolated
    ``run_window`` output's ``equity_records`` (one entry per recorded bar,
    in order) -- 0 everywhere outside the recorded range.

    Takes the whole ``run`` rather than its records alone so the absolute
    bar of entry 0 comes from that same run's ``effective_start``. That is
    the window's start only when the run's pad covered the strategy's
    warmup; assuming it unconditionally would shift every flag earlier by
    the shortfall on runs where it did not.
    """
    flags = {sym: np.zeros(n_bars, dtype='float64') for sym in symbols}
    for j, rec in enumerate(run['result']['equity_records']):
        bar = run['effective_start'] + j
        if bar >= n_bars:
            break
        for sym, pos in rec['position'].items():
            if sym in flags:
                flags[sym][bar] = 1.0 if pos != 0 else 0.0
    return flags


# ---------------------------------------------------------------------
# Purged walk-forward
# ---------------------------------------------------------------------

def purge_events(events: list, train_window: Window, valid_window: Window, embargo: int = 10) -> tuple:
    """Split ``events`` into ``(train_events, valid_events)`` for one fold.

    ``valid_events``: signal fired inside ``valid_window``.
    ``train_events``: signal fired inside ``train_window`` AND whose
    ``[open_bar, close_bar]`` holding interval does not overlap
    ``[valid_window.start - embargo, valid_window.end + embargo)`` -- an
    event whose outcome depended on price action in or near the validation
    window is purged rather than kept, however tempting its label is.
    """
    valid = [e for e in events if valid_window.start <= e['signal_bar'] < valid_window.end]
    lo, hi = valid_window.start - embargo, valid_window.end + embargo

    train = []
    for e in events:
        if not (train_window.start <= e['signal_bar'] < train_window.end):
            continue
        close_bar = e['close_bar'] if e['close_bar'] is not None else e['open_bar']
        if e['open_bar'] < hi and close_bar >= lo:
            continue  # purge: holding interval overlaps the embargoed valid window
        train.append(e)
    return train, valid


# ---------------------------------------------------------------------
# Dataset / model
# ---------------------------------------------------------------------

def build_dataset(events: list, frames: dict, feature_names: list = None) -> tuple:
    rows = [frames[e['symbol']].loc[e['signal_bar']] for e in events]
    X = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
    if feature_names is not None:
        X = X.reindex(columns=feature_names)
    y = np.array([1 if e['label'] else 0 for e in events], dtype='int64')
    w = np.array([e['weight'] for e in events], dtype='float64')
    if w.sum() > 0:
        w = w / w.mean()  # normalize around 1.0 so it combines sensibly with class_weight
    else:
        w = np.ones_like(w)
    return X, y, w


def select_features(X: pd.DataFrame, y: np.ndarray, max_features: int = 12, seed: int = 0) -> list:
    if X.shape[1] <= max_features:
        return list(X.columns)
    filled = X.fillna(X.median(numeric_only=True))
    mi = mutual_info_classif(filled, y, random_state=seed)
    order = np.argsort(mi)[::-1]
    return [X.columns[i] for i in order[:max_features]]


def fit_model(X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray, kind: str = 'logit', seed: int = 0) -> Pipeline:
    if len(np.unique(y)) < 2:
        raise ValueError('Training fold has only one class -- cannot fit a classifier.')
    pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', MODEL_REGISTRY[kind](seed=seed)),
    ])
    pipeline.fit(X, y, clf__sample_weight=sample_weight)
    return pipeline


def oof_predict(events: list, frames: dict, folds: list, *, kind='logit', embargo=10, max_features=12, seed=0, min_train_events=20) -> np.ndarray:
    """Out-of-fold ``P(win)`` for every event that falls in some fold's
    valid window, ``nan`` for events that don't (e.g. inside an embargo gap
    or before the first fold's valid window begins).
    """
    oof = np.full(len(events), np.nan)
    index_of = {id(e): i for i, e in enumerate(events)}

    for train_window, valid_window in folds:
        train_events, valid_events = purge_events(events, train_window, valid_window, embargo=embargo)
        if len(train_events) < min_train_events or len(valid_events) == 0:
            continue
        X_train, y_train, w_train = build_dataset(train_events, frames)
        if len(set(y_train)) < 2:
            logger.warning('Fold %s: training events are single-class after purging -- skipped.', valid_window.name)
            continue
        feats = select_features(X_train, y_train, max_features=max_features, seed=seed)
        pipeline = fit_model(X_train[feats], y_train, w_train, kind=kind, seed=seed)
        X_valid, _, _ = build_dataset(valid_events, frames, feature_names=feats)
        proba = pipeline.predict_proba(X_valid)[:, 1]
        for e, p in zip(valid_events, proba):
            oof[index_of[id(e)]] = p

    return oof


def select_threshold(events: list, oof_proba: np.ndarray, thresholds=None, min_keep_frac: float = 0.4) -> dict:
    """Pick the threshold maximizing ``expectancy(kept trades) *
    sqrt(n_kept)`` over out-of-fold predictions -- expected value, not
    accuracy/AUC, is exactly what should decide this for a strategy whose
    profit is concentrated in a few big winners.
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.02)
    n = len(events)

    def _score(idx):
        pnls = [events[i]['net_pnl'] for i in idx]
        ev = float(np.mean(pnls)) if pnls else float('-inf')
        return ev, ev * math.sqrt(len(idx))

    best = None
    for t in thresholds:
        idx = [i for i in range(n) if oof_proba[i] >= t]
        if n and len(idx) / n < min_keep_frac:
            continue
        ev, objective = _score(idx)
        if best is None or objective > best['objective']:
            best = {'threshold': float(t), 'objective': objective, 'expectancy': ev,
                     'n_kept': len(idx), 'keep_frac': len(idx) / n if n else 0.0}

    if best is None:
        # Every candidate threshold would keep less than min_keep_frac of the
        # events -- fall back to keeping everything rather than erroring. The
        # threshold has to be 0.0 here, not min(thresholds): it is what ends up
        # in the deployed bundle and gates live entries, so any value above the
        # lowest observed probability would silently contradict the n_kept /
        # keep_frac reported next to it (and could ship a filter that blocks
        # every trade while claiming a 100% keep rate).
        idx = list(range(n))
        ev, objective = _score(idx)
        best = {'threshold': 0.0, 'objective': objective, 'expectancy': ev,
                'n_kept': n, 'keep_frac': 1.0}
    return best


# ---------------------------------------------------------------------
# End-to-end fit
# ---------------------------------------------------------------------

def fit_meta_model(
    market: MarketData,
    strategy_cls: type,
    params: dict,
    *,
    cash: float,
    slippage: float = 0.0,
    n_folds: int = 4,
    embargo: int = 10,
    holdout_frac: float = 0.20,
    kind: str = 'logit',
    max_features: int = 12,
    seed: int = 42,
    min_events_per_fold: int = 20,
    reserve_bars: int = None,
) -> dict:
    """Full pipeline: extract events over the non-holdout span, run purged
    walk-forward for out-of-fold probabilities, pick a threshold from those,
    then refit on every usable event for deployment. Returns a bundle dict
    ready for ``joblib.dump``.
    """
    from research.warmup import probe_warmup  # local import: avoids a cycle with research.optimize
    from strategies import name_for

    if reserve_bars is None:
        reserve_bars = max(probe_warmup(market, strategy_cls, params), 252) + 10

    folds, holdout = anchored_walk_forward(
        market.n_bars, reserve_bars=reserve_bars, n_folds=n_folds,
        embargo=embargo, holdout_frac=holdout_frac,
    )
    full_window = Window('full', reserve_bars, holdout.start)

    extraction = extract_events(market, strategy_cls, params, full_window, cash=cash, slippage=slippage, pad=reserve_bars)
    raw_events = extraction['events']
    events = [e for e in raw_events if not e['forced'] and not e['open_at_end']]

    if len(events) < min_events_per_fold * n_folds:
        raise ValueError(
            f"Only {len(events)} usable signal events over the non-holdout span -- too "
            f"few to fit a meta-label model with {n_folds} folds (need >= "
            f"{min_events_per_fold} per fold). This strategy trades too rarely for "
            f"meta-labeling to have enough data; skipping is safer than fitting on noise."
        )

    flags = position_flags(market.n_bars, extraction['run'], market.symbols)
    frames = build_feature_matrix(market, strategy_cls, params, events=events, position_flags=flags)

    oof = oof_predict(events, frames, folds, kind=kind, embargo=embargo, max_features=max_features, seed=seed)
    scored_idx = [i for i in range(len(events)) if not np.isnan(oof[i])]
    if len(scored_idx) < min_events_per_fold:
        raise ValueError(
            f"Only {len(scored_idx)} events received an out-of-fold prediction -- not "
            f"enough to select a threshold reliably."
        )
    oof_events = [events[i] for i in scored_idx]
    oof_valid = oof[scored_idx]
    threshold_info = select_threshold(oof_events, oof_valid)

    y_oof = np.array([1 if e['label'] else 0 for e in oof_events])
    try:
        auc = float(roc_auc_score(y_oof, oof_valid)) if len(set(y_oof)) > 1 else float('nan')
    except ValueError:
        auc = float('nan')

    # Refit on every usable event for deployment. No leakage risk here: fold
    # discipline above governed *selection* (features, threshold); once
    # that's decided there is no more held-out data within this span to
    # protect -- the holdout window itself is still never touched.
    X_all, y_all, w_all = build_dataset(events, frames)
    final_features = select_features(X_all, y_all, max_features=max_features, seed=seed)
    final_pipeline = fit_model(X_all[final_features], y_all, w_all, kind=kind, seed=seed)

    return {
        'pipeline': final_pipeline,
        'features': final_features,
        'threshold': threshold_info['threshold'],
        'threshold_info': threshold_info,
        'strategy_key': name_for(strategy_cls),
        'strategy_class': strategy_cls.__name__,
        'params': params,
        'symbols': sorted(market.symbols),
        'cash': cash,
        'slippage': slippage,
        'reserve_bars': reserve_bars,
        'n_folds': n_folds,
        'embargo': embargo,
        'holdout_frac': holdout_frac,
        'full_window': {'start': full_window.start, 'end': full_window.end},
        'kind': kind,
        'seed': seed,
        'max_features': max_features,
        'n_events_total': len(raw_events),
        'n_events_usable': len(events),
        'n_events_oof': len(oof_events),
        'oof_auc': auc,
    }


# ---------------------------------------------------------------------
# Evaluation: gated vs. baseline vs. random-rejection
# ---------------------------------------------------------------------

def _run_gated(market, strategy_cls, params, proba, threshold, window, *,
               cash, slippage, pad, on_missing='block'):
    """``proba``'s arrays are indexed by absolute bar, but the engine inside
    ``run_window`` only sees the ``[slice_start(window, pad), window.end)``
    slice and counts bars from 0 -- so the gate needs that slice origin to
    read the right cell. Derived from the same helper ``run_window`` uses, so
    the two can't drift apart.
    """
    bar_offset = slice_start(window, pad)

    def factory(**_ignored):
        return MetaFilteredStrategy(
            strategy_cls(**params), proba, bar_offset=bar_offset,
            on_missing=on_missing, meta_threshold=threshold,
        )
    return run_window(market, factory, {}, window, cash=cash, slippage=slippage, pad=pad)


def _slice_metrics(run: dict, window: Window, cash: float) -> dict:
    """Metrics for the ``window`` slice of a ``run_window`` output that was
    actually run over a wider window (both starting flat at that wider
    window's start).

    Used instead of a fresh isolated re-run scoped to ``window`` alone,
    because a strategy that's always in the market (e.g. a crossover that
    is never flat) would otherwise force an artificial fresh entry right at
    ``window.start`` that has no counterpart in the continuous run
    ``extract_events`` used to build the events/probabilities in the first
    place -- comparing the two would be comparing different trajectories,
    not the effect of gating on the *same* one.

    Trades are attributed to the window that contains their ``close_bar``
    (when the P&L is actually booked), not ``open_bar``. A position that
    straddles the window boundary can still make ``compute_metrics`` log a
    reconciliation-drift warning here (its mark-to-market path crosses the
    boundary even though its trade row doesn't) -- harmless for this
    function's purpose, since the ``sharpe_ratio`` and ``max_drawdown`` this
    is actually used for come from the equity slice directly, not from
    trade_logs reconciling against it.
    """
    # Offsets are taken against the run's own first *recorded* bar, not the
    # start of the window it was asked for: a pad too small for the
    # strategy's warmup makes the engine drop the un-tradeable head bars
    # from the curve, and slicing as if they were present would read the
    # wrong stretch of equity for ``window``.
    origin = run['effective_start']
    result = run['result']
    a = window.start - origin
    b = window.end - origin
    sliced_equity = result['equity_records'][max(a, 0):b]
    trades = [
        t for t in result['trade_logs']
        if t['close_bar'] is not None and window.start <= t['close_bar'] < window.end
    ]
    liquidations = sum(1 for t in trades if t['forced'])
    return compute_metrics(sliced_equity, trades, cash, liquidation_count=liquidations)


def evaluate_meta_backtest(bundle: dict, market: MarketData, *, n_random: int = 200, seed: int = 0) -> dict:
    """Compare a gated backtest against the ungated baseline over the union
    of walk-forward valid windows, plus a random-rejection baseline of the
    same size.

    Gating here uses freshly-recomputed *out-of-fold* probabilities (each
    fold's model trained only on its own purged training events), not
    ``bundle['pipeline']`` (which was refit on every usable event and would
    leak into any window this function could evaluate on). This is
    therefore a second, independent re-derivation of the OOF predictions
    ``fit_meta_model`` used to pick the threshold -- an honest check, not a
    replay of the same numbers.

    Both the gated arm and the random arm run with ``on_missing='pass'``: an
    event only receives an out-of-fold probability if its signal bar falls in
    some fold's *valid* window, so most bars carry no prediction for reasons
    that have nothing to do with the market. Blocking those would charge the
    filter for the harness's coverage gaps, and gated-vs-baseline would then
    measure "how many bars had an OOF prediction" rather than "what did the
    filter decide". Letting them through makes the gated arm diverge from the
    baseline *only* where the model actually had an opinion -- and since every
    scored event lies inside ``eval_window`` by construction, the two arms are
    also bar-for-bar identical up to ``eval_window.start``, so they enter the
    comparison in the same position state.

    ``oof_coverage`` reports how much of the evaluation window the filter
    actually got to speak on. A low value does not bias the comparison, but it
    does mean the verdict rests on few decisions -- read it before the Sharpes.
    """
    strategy_cls = load_strategy(bundle['strategy_key'])
    params = bundle['params']
    reserve_bars = bundle['reserve_bars']
    cash, slippage = bundle['cash'], bundle['slippage']

    folds, holdout = anchored_walk_forward(
        market.n_bars, reserve_bars=reserve_bars, n_folds=bundle['n_folds'],
        embargo=bundle['embargo'], holdout_frac=bundle['holdout_frac'],
    )
    full_window = Window('full', reserve_bars, holdout.start)

    extraction = extract_events(market, strategy_cls, params, full_window, cash=cash, slippage=slippage, pad=reserve_bars)
    events = [e for e in extraction['events'] if not e['forced'] and not e['open_at_end']]
    flags = position_flags(market.n_bars, extraction['run'], market.symbols)
    frames = build_feature_matrix(market, strategy_cls, params, events=events, position_flags=flags)

    oof = oof_predict(
        events, frames, folds, kind=bundle['kind'], embargo=bundle['embargo'],
        max_features=bundle['max_features'], seed=bundle['seed'],
    )
    scored = [i for i in range(len(events)) if not np.isnan(oof[i])]
    if not scored:
        raise ValueError('No out-of-fold predictions available for meta-backtest evaluation.')

    eval_window = Window('oof_eval', folds[0][1].start, folds[-1][1].end)
    threshold = bundle['threshold']

    scored_events = [events[i] for i in scored]
    scored_proba = oof[scored]
    proba_by_symbol = {sym: np.full(market.n_bars, np.nan) for sym in market.symbols}
    for e, p in zip(scored_events, scored_proba):
        proba_by_symbol[e['symbol']][e['signal_bar']] = p

    gated_full = _run_gated(
        market, strategy_cls, params, proba_by_symbol, threshold, full_window,
        cash=cash, slippage=slippage, pad=reserve_bars, on_missing='pass',
    )
    base_out = _slice_metrics(extraction['run'], eval_window, cash)
    gated_out_metrics = _slice_metrics(gated_full, eval_window, cash)

    gate = gated_full['engine'].strategy
    gate_counters = {
        'threshold_rejected': gate.threshold_rejected_count,
        'blocked_total': gate.blocked_count,
        'no_prediction_bars': gate.nan_count + gate.out_of_range_count,
    }

    rejected_events = [e for e, p in zip(scored_events, scored_proba) if p < threshold]
    kept_events = [e for e, p in zip(scored_events, scored_proba) if p >= threshold]
    rejected_net_pnl = float(sum(e['net_pnl'] for e in rejected_events))

    in_window = [e for e in events if eval_window.start <= e['signal_bar'] < eval_window.end]
    coverage = len(scored_events) / len(in_window) if in_window else float('nan')

    n_reject = len(rejected_events)
    rng = np.random.default_rng(seed)
    random_sharpes = []
    for _ in range(n_random):
        reject_idx = set(rng.choice(len(scored_events), size=min(n_reject, len(scored_events)), replace=False)) if n_reject else set()
        rand_proba = {sym: np.full(market.n_bars, np.nan) for sym in market.symbols}
        for j, e in enumerate(scored_events):
            rand_proba[e['symbol']][e['signal_bar']] = 0.0 if j in reject_idx else 1.0
        rand_full = _run_gated(
            market, strategy_cls, params, rand_proba, 0.5, full_window,
            cash=cash, slippage=slippage, pad=reserve_bars, on_missing='pass',
        )
        rand_metrics = _slice_metrics(rand_full, eval_window, cash)
        random_sharpes.append(rand_metrics.get('sharpe_ratio', 0.0))

    gated_sharpe = gated_out_metrics.get('sharpe_ratio', 0.0)
    if n_reject and random_sharpes:
        percentile = float(np.mean(np.array(random_sharpes) <= gated_sharpe) * 100.0)
        beats_random = percentile >= 90.0
    else:
        # The filter rejected nothing, so every "random rejection of the same
        # size" is the gated run itself: the distribution is a point mass and a
        # percentile of it would be 100 by construction, not by merit. Say
        # there is no comparison rather than manufacture a passing grade.
        percentile = None
        beats_random = None

    return {
        'window': {'start': eval_window.start, 'end': eval_window.end},
        'baseline_metrics': base_out,
        'gated_metrics': gated_out_metrics,
        'n_events_scored': len(scored_events),
        'n_rejected': n_reject,
        'n_kept': len(kept_events),
        'rejected_net_pnl': rejected_net_pnl,
        'rejected_net_pnl_positive': rejected_net_pnl > 0,
        'n_events_in_window': len(in_window),
        'oof_coverage': coverage,
        'gate_counters': gate_counters,
        'random_baseline': {
            'n_random': n_random,
            'sharpe_mean': float(np.mean(random_sharpes)) if random_sharpes else float('nan'),
            'sharpe_std': float(np.std(random_sharpes)) if random_sharpes else float('nan'),
            'gated_sharpe': gated_sharpe,
            'gated_percentile': percentile,
            'beats_random': beats_random,
            'degenerate': n_reject == 0,
        },
        'oof_auc': bundle.get('oof_auc'),
    }
