"""Meta-labeling: a second model that vetoes the primary strategy's entries.

The primary strategy (any ``strategies.base.Strategy``) decides *direction*
and is left untouched. A binary classifier then judges each entry it proposes
-- "will this trade make money, net of costs, under this strategy's own exits"
-- and suppresses the ones below a threshold. It can only subtract: no new
entries, no direction changes, and never a blocked exit.

Layout:

    features.py   one feature definition, shared by training and inference
    dataset.py    trade log -> (X, y, w), plus the purge rule
    model.py      fitting, thresholds, and the pass-through/walk-forward/live
                  model shapes the gate is agnostic to
    filter.py     make_meta_filtered(cls, model) -> a gated Strategy subclass
    evaluate.py   AUC, precision lift, and the shuffled-label control

Driven by ``meta_runner.py`` (harvest / walkforward / fit / signal).
"""

from __future__ import annotations

from meta.dataset import Samples, purged_train_mask, samples_from_trades, window_mask
from meta.features import FEATURE_NAMES, build_feature_arrays, feature_row
from meta.filter import make_meta_filtered, split_order, unfiltered
from meta.model import (
    FittedMetaModel,
    PassThroughModel,
    WalkForwardMetaModel,
    fit_estimator,
    load_model,
    save_model,
    threshold_for,
)

__all__ = [
    'FEATURE_NAMES', 'build_feature_arrays', 'feature_row',
    'Samples', 'samples_from_trades', 'purged_train_mask', 'window_mask',
    'make_meta_filtered', 'unfiltered', 'split_order',
    'PassThroughModel', 'FittedMetaModel', 'WalkForwardMetaModel',
    'fit_estimator', 'threshold_for', 'save_model', 'load_model',
]
