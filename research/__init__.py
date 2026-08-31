"""Research tools: Optuna parameter optimization with walk-forward validation.

Everything in this package is strategy-agnostic -- it operates on any
``strategies.base.Strategy`` subclass through its declared or inferred
``space`` (see :mod:`research.space`) and through the generic
``BarContext``/trade-log interfaces already exposed by ``core``. Nothing
here should ever import a concrete strategy class by name.
"""
