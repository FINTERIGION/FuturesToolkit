"""Strategies package: base class + public examples.

Put private research modules in this folder (gitignored). They are picked
up automatically by :func:`discover_strategies` -- no registration needed.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

from .base import BarContext, SetupContext, Strategy
from .double_ma import DoubleMaStrategy
from .my_strategy import MyStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy

__all__ = [
    'Strategy', 'SetupContext', 'BarContext',
    'DoubleMaStrategy', 'RsiMeanReversionStrategy', 'MyStrategy',
    'discover_strategies', 'load_strategy', 'name_for',
]

_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')


def name_for(cls: type) -> str:
    """``DoubleMaStrategy`` -> ``'double_ma'``: snake_case the class name,
    then drop a redundant trailing ``_strategy`` -- unless doing so would
    leave a single, uninformative word (``MyStrategy`` -> ``'my_strategy'``,
    not ``'my'``). Matches the CLI names already documented in README.md."""
    snake = _CAMEL_RE.sub('_', cls.__name__).lower()
    if snake.endswith('_strategy'):
        stem = snake[: -len('_strategy')]
        if '_' in stem:
            return stem
    return snake


def discover_strategies() -> dict:
    """Scan every module in this package and return ``{short_name: cls}`` for
    every concrete ``Strategy`` subclass found (``Strategy`` itself excluded).

    A strategy is included regardless of which module defines it, so long as
    it lives somewhere under ``strategies/`` -- private, gitignored modules
    are discovered the same as the bundled examples.
    """
    found: dict = {}
    package = importlib.import_module(__name__)
    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(package.__path__, prefix=f'{__name__}.'):
        module = importlib.import_module(mod_name)
        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Strategy:
                continue
            if not issubclass(obj, Strategy):
                continue
            if obj.__module__ != module.__name__:
                continue  # re-exported import, not defined here
            found[name_for(obj)] = obj
    return found


def load_strategy(spec: str) -> type:
    """Resolve ``spec`` to a ``Strategy`` subclass.

    ``spec`` is either a short name from :func:`discover_strategies` (e.g.
    ``'double_ma'``) or a ``'module.path:ClassName'`` reference to a strategy
    living outside this package.
    """
    if ':' in spec:
        module_name, class_name = spec.split(':', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if not (inspect.isclass(cls) and issubclass(cls, Strategy)):
            raise TypeError(f"{spec!r} is not a Strategy subclass")
        return cls
    registry = discover_strategies()
    try:
        return registry[spec]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {spec!r}. Available: {', '.join(sorted(registry))}"
        ) from None
