"""Research tools: overfitting checks over walk-forward and sub-period splits.

Everything in this package is strategy-agnostic -- it operates on any
``strategies.base.Strategy`` subclass through its declared or inferred
parameter ranges (see :mod:`research.space`) and through the generic
``BarContext``/trade-log interfaces already exposed by ``core``. Nothing
here should ever import a concrete strategy class by name.

Nothing here searches for parameters either. It used to: an Optuna study
picked a winner off these same folds, which on a short history of correlated
products reliably found the luckiest configuration rather than an edge. The
diagnostics that study computed afterwards were the part worth keeping, and
:mod:`research.validate` now runs them against whatever parameters you bring.
"""
