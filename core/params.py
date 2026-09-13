"""Declarative parameter-range types.

Deliberately dependency-free so ``strategies/`` never has to pull in the
research stack just to say what range a parameter is plausible over. A
``Strategy`` subclass declares that with a class-level
``space: dict[str, Int | Float | Categorical]`` alongside its existing
``params`` defaults. Nothing searches those ranges: ``research.validate``
steps one notch either side of the value in use to check the result does not
hinge on the exact number, and the web panel's parameter editor uses the same
bounds to flag a value typed outside them.
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
