"""Factors package: base class + bundled examples.

Put private research factors in this folder (gitignored, same convention as
``strategies/``). They are picked up automatically by
:func:`discover_factors` -- no registration needed. ``factor_runner.py``
judges a factor directly (IC, quantile buckets, decay -- see
``research.factor_eval``); ``strategies.factor_bridge`` turns any discovered
factor into a runnable, tunable ``Strategy`` for free.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

from .base import Factor, FactorContext, FactorPanel, compute_factor

__all__ = [
    'Factor', 'FactorContext', 'FactorPanel', 'compute_factor',
    'discover_factors', 'load_factor', 'name_for',
]

_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')


def name_for(cls: type) -> str:
    """``CarryFactor`` -> ``'carry'``: snake_case the class name, then drop a
    redundant trailing ``_factor`` -- unless doing so would leave nothing at
    all, in which case the unstripped name is kept."""
    snake = _CAMEL_RE.sub('_', cls.__name__).lower()
    if snake.endswith('_factor'):
        stem = snake[: -len('_factor')]
        if stem:
            return stem
    return snake


def discover_factors() -> dict:
    """Scan every module in this package and return ``{short_name: cls}`` for
    every concrete ``Factor`` subclass found (``Factor`` itself excluded).

    A factor is included regardless of which module defines it, so long as
    it lives somewhere under ``factors/`` -- private, gitignored modules are
    discovered the same as the bundled examples.
    """
    found: dict = {}
    package = importlib.import_module(__name__)
    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(package.__path__, prefix=f'{__name__}.'):
        module = importlib.import_module(mod_name)
        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Factor:
                continue
            if not issubclass(obj, Factor):
                continue
            if obj.__module__ != module.__name__:
                continue  # re-exported import, not defined here
            found[name_for(obj)] = obj
    return found


def load_factor(spec: str) -> type:
    """Resolve ``spec`` to a ``Factor`` subclass.

    ``spec`` is either a short name from :func:`discover_factors` (e.g.
    ``'carry'``) or a ``'module.path:ClassName'`` reference to a factor
    living outside this package.
    """
    if ':' in spec:
        module_name, class_name = spec.split(':', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not (inspect.isclass(cls) and issubclass(cls, Factor)):
            raise TypeError(f"{spec!r} is not a Factor subclass")
        return cls
    registry = discover_factors()
    try:
        return registry[spec]
    except KeyError:
        raise KeyError(
            f"Unknown factor {spec!r}. Available: {', '.join(sorted(registry))}"
        ) from None
