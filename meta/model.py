"""Meta-model fitting, thresholds, and the three shapes a gate can be handed.

``meta.filter`` never learns which of these it is holding. That is the point:
the walk-forward backtest and the live signal run *identical* gate code, and
only the object answering ``proba`` differs. A bug that only shows up in one
of the two paths therefore has nowhere to live.

    PassThroughModel     proba -> None            everything is allowed
    WalkForwardMetaModel one estimator per fold   backtest, dispatched by bar
    FittedMetaModel      one estimator            live

**Why the model is queried per bar instead of pre-computing a probability
table.** Filtering changes which bars the primary strategy can enter on: veto
an entry and the product stays flat, so the next breakout two days later --
a bar that never produced a trade in the unfiltered run -- is now a live
entry. A lookup table built from the unfiltered trade log has no row for it.
The gate has to be able to score any (symbol, bar, side) on demand, so the
model holds fitted estimators and features get computed as the run goes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date as Date
from typing import List, Optional, Sequence, Tuple

import numpy as np

from meta.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    'PassThroughModel', 'FittedMetaModel', 'WalkForwardMetaModel',
    'fit_estimator', 'threshold_for', 'save_model', 'load_model',
]


class PassThroughModel:
    """Allows every order. The unfiltered baseline, run through the *same*
    wrapper as the filtered arm so both register the same feature indicators
    and therefore start on the same bar -- see ``meta.filter`` for why an
    unwrapped baseline would not be a fair comparison.
    """

    kind = 'passthrough'

    def proba(self, sym: str, when: Date, x: np.ndarray) -> Optional[float]:
        return None

    def threshold(self, when: Date) -> float:
        return 0.0


@dataclass
class FittedMetaModel:
    """A single fitted estimator with a single threshold. Live, and holdout."""

    estimator: object
    threshold_value: float
    keep_rate: float
    kind: str = 'rf'
    feature_names: Sequence[str] = tuple(FEATURE_NAMES)
    trained_through: Optional[str] = None   # ISO date of the last training bar

    def proba(self, sym: str, when: Date, x: np.ndarray) -> Optional[float]:
        return float(self.estimator.predict_proba(x.reshape(1, -1))[0, 1])

    def threshold(self, when: Date) -> float:
        return self.threshold_value


class WalkForwardMetaModel:
    """Per-fold estimators, dispatched on **date**.

    Each entry is ``(first_date, last_date, estimator, threshold)``, both ends
    inclusive, where the estimator was fitted only on trades that *closed*
    before that fold's train window ended.

    **Dates, not bar indices.** ``research.runner_api.run_window`` evaluates a
    fold on a *slice* of the market that starts ``pad`` bars before the window,
    and the engine inside it counts bars from zero on that slice -- so
    ``ctx.i`` is off by ``lo`` from the fold boundaries, which are expressed in
    the full market's numbering. Dispatching on bar index silently matched
    nothing and let every order through; the filtered arm then scored exactly
    the same as the baseline no matter what the threshold was. A date means
    the same thing in every slice, so this cannot recur.

    A date outside every validation window -- the first train chunk, or an
    embargo gap -- gets ``None``, which the gate reads as "no opinion, let it
    through". Fail-open rather than fail-closed so that "the model had no
    coverage here" and "the model rejected this" stay distinguishable in the
    results instead of both showing up as a missing trade.
    """

    kind = 'walkforward'

    def __init__(self, folds: List[Tuple[Date, Date, object, float]]):
        self._folds = sorted(folds, key=lambda f: f[0])

    def _fold_for(self, when: Date):
        for first, last, est, thr in self._folds:
            if first <= when <= last:
                return est, thr
        return None

    def proba(self, sym: str, when: Date, x: np.ndarray) -> Optional[float]:
        fold = self._fold_for(when)
        if fold is None:
            return None
        return float(fold[0].predict_proba(x.reshape(1, -1))[0, 1])

    def threshold(self, when: Date) -> float:
        fold = self._fold_for(when)
        return fold[1] if fold is not None else 0.0

    @property
    def covered_dates(self) -> List[Tuple[Date, Date]]:
        return [(f[0], f[1]) for f in self._folds]


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------

def fit_estimator(X: np.ndarray, y: np.ndarray, w: np.ndarray, *, kind: str = 'rf', seed: int = 42):
    """Fit one binary classifier.

    Both configurations are deliberately small. A daily strategy on this
    universe yields order-1k trades against ~15 features; ``max_depth=3`` and
    ``min_samples_leaf=25`` are the main defence against the model memorising
    that, and they matter far more than which family gets picked. ``lr`` is
    not a fallback -- it is the baseline that ``rf`` has to beat before a tree
    is worth using at all.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2:
        raise ValueError(
            f'meta-model needs both classes to fit; got only y={np.unique(y).tolist()} '
            f'over {len(y)} samples'
        )

    if kind == 'rf':
        est = RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=25,
            class_weight='balanced_subsample', random_state=seed, n_jobs=-1,
        )
    elif kind == 'lr':
        est = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, random_state=seed),
        )
    else:
        raise ValueError(f"unknown estimator kind {kind!r}; expected 'rf' or 'lr'")

    est.fit(X, y, **({} if kind == 'lr' else {'sample_weight': w}))
    return est


def threshold_for(estimator, X_train: np.ndarray, keep_rate: float) -> float:
    """Probability cut that would have kept ``keep_rate`` of the *training* trades.

    Chosen on train only, never on the validation window it is then applied
    to. That makes ``keep_rate`` -- a round number fixed in advance, reported
    as a curve over {1.0, 0.8, 0.65, 0.5} -- the single human constant in the
    procedure, rather than a threshold tuned against the very curve it is
    supposed to be judged by. ``keep_rate=1.0`` returns a cut below every
    training probability, i.e. an exact no-op, which is what makes it usable
    as a self-check.
    """
    if not (0.0 < keep_rate <= 1.0):
        raise ValueError(f'keep_rate must be in (0, 1]; got {keep_rate}')
    p = estimator.predict_proba(X_train)[:, 1]
    if keep_rate >= 1.0:
        return float(np.min(p)) - 1.0
    return float(np.quantile(p, 1.0 - keep_rate))


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def save_model(model: FittedMetaModel, path: str, *, meta: dict) -> str:
    """Write ``path`` (joblib) plus ``path + '.json'`` (human-readable sidecar).

    The sidecar carries everything needed to tell whether the artifact still
    applies: feature names and order, the primary strategy and its params, the
    training window, and library versions.
    """
    import joblib
    import sklearn

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    joblib.dump(model, path)
    sidecar = {
        'feature_names': list(model.feature_names),
        'kind': model.kind,
        'keep_rate': model.keep_rate,
        'threshold': model.threshold_value,
        'trained_through': model.trained_through,
        'sklearn_version': sklearn.__version__,
        **meta,
    }
    with open(path + '.json', 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False, default=str)
    logger.info('Meta-model saved: %s (+ .json)', path)
    return path


def load_model(path: str) -> FittedMetaModel:
    """Load a saved model, refusing one whose feature contract has drifted.

    A reordered or renamed feature does not raise on its own -- the estimator
    happily scores a vector of the right width -- it just returns confident
    nonsense. Checking the stored names against the live ``FEATURE_NAMES``
    turns that into an error at load time, which is the only place it is
    cheap to catch.
    """
    import joblib

    model = joblib.load(path)
    stored = list(model.feature_names)
    if stored != list(FEATURE_NAMES):
        raise ValueError(
            f'{path} was trained on a different feature set and cannot be used.\n'
            f'  stored: {stored}\n'
            f'  current: {list(FEATURE_NAMES)}\n'
            'Re-run `meta_runner.py fit` against the current meta.features.'
        )
    return model
