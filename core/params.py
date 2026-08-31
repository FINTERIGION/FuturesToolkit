"""Declarative parameter-space types for strategy tuning.

Deliberately dependency-free (no Optuna import here) so ``strategies/`` never
has to pull in the research stack just to declare a search space. A
``Strategy`` subclass declares its tunable surface with a class-level
``space: dict[str, Int | Float | Categorical]`` alongside its existing
``params`` defaults; ``research.space`` turns that into Optuna suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union


@dataclass(frozen=True)
class Int:
    low: int
    high: int
    step: int = 1
    log: bool = False


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    step: float | None = None
    log: bool = False


@dataclass(frozen=True)
class Categorical:
    choices: Tuple[object, ...]

    def __init__(self, choices):
        object.__setattr__(self, 'choices', tuple(choices))


Spec = Union[Int, Float, Categorical]
