"""Cross-factor correlation: is a new factor actually saying something new?

Two different questions, and they can disagree:

``correlation_matrix``
    Do two factors *rank the products the same way*? Measured per bar across
    the cross-section, then averaged. Two factors at 0.9 here are largely the
    same bet, and combining them mostly doubles down rather than diversifying.

``ic_correlation``
    Do two factors *work at the same times*? Measured between their IC series
    over the calendar. This is the one that decides whether combining helps:
    factors can rank products similarly yet earn in different regimes (low
    IC correlation, worth combining), or rank differently yet win and lose
    together (high IC correlation, less diversification than the first matrix
    suggests).

Pure functions over arrays and ``FactorPanel``s; nothing here imports a
concrete factor.
"""

from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np

from factors.base import FactorPanel
from research.factor_eval import joint_ranks, rowwise_corr

__all__ = ['correlation_matrix', 'ic_correlation', 'to_rows']


def _pairwise_cs_corr(a: np.ndarray, b: np.ndarray, method: str = 'spearman', min_names: int = 3) -> float:
    if method == 'spearman':
        ra, rb = joint_ranks(a, b)
        corr = rowwise_corr(ra, rb, min_names)
    elif method == 'pearson':
        corr = rowwise_corr(a, b, min_names)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'spearman' or 'pearson'")
    valid = corr[~np.isnan(corr)]
    return float(valid.mean()) if valid.size else float('nan')


def correlation_matrix(panels: Mapping[str, FactorPanel], method: str = 'spearman', min_names: int = 3) -> dict:
    """Average cross-sectional correlation between every pair of factors.

    Correlating each bar's cross-section and then averaging -- rather than
    pooling every (bar, product) cell into one big correlation -- keeps a
    period of unusually wide dispersion from dominating the answer, and
    matches how the factors are actually used: one ranking per bar.

    Returns ``{'names': [...], 'method': ..., 'matrix': [[...]]}`` with the
    matrix in ``names`` order.
    """
    names = list(panels)
    if len(names) < 2:
        raise ValueError("correlation_matrix: need at least 2 factors")
    shape = panels[names[0]].values.shape
    for name in names[1:]:
        if panels[name].values.shape != shape:
            raise ValueError(
                f"correlation_matrix: {name} has shape {panels[name].values.shape}, "
                f"expected {shape} -- factors must be computed over the same window and universe"
            )

    n = len(names)
    matrix = np.full((n, n), np.nan, dtype='float64')
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            c = _pairwise_cs_corr(panels[names[i]].values, panels[names[j]].values, method, min_names)
            matrix[i, j] = c
            matrix[j, i] = c
    return {'names': names, 'method': method, 'matrix': matrix}


def ic_correlation(ics: Mapping[str, np.ndarray], min_names: int = 3) -> dict:
    """Correlation between factors' IC time series, over bars they both scored.

    High values mean the factors succeed and fail together, so holding both
    diversifies less than their cross-sectional correlation implies.

    Returns ``{'names': [...], 'matrix': [[...]]}`` with the matrix in
    ``names`` order.
    """
    names = list(ics)
    if len(names) < 2:
        raise ValueError("ic_correlation: need at least 2 IC series")
    shape = np.asarray(ics[names[0]]).shape
    for name in names[1:]:
        if np.asarray(ics[name]).shape != shape:
            raise ValueError(
                f"ic_correlation: {name} has shape {np.asarray(ics[name]).shape}, expected {shape}"
            )

    n = len(names)
    matrix = np.full((n, n), np.nan, dtype='float64')
    for i in range(n):
        matrix[i, i] = 1.0
        ai = np.asarray(ics[names[i]], dtype='float64').reshape(1, -1)
        for j in range(i + 1, n):
            bj = np.asarray(ics[names[j]], dtype='float64').reshape(1, -1)
            c = float(rowwise_corr(ai, bj, min_names)[0])
            matrix[i, j] = c
            matrix[j, i] = c
    return {'names': names, 'matrix': matrix}


def to_rows(result: dict) -> List[Dict[str, object]]:
    """Flatten a correlation result into JSON-friendly ``{'factor': name, ...}`` rows."""
    names = result['names']
    matrix = np.asarray(result['matrix'], dtype='float64')
    rows: List[Dict[str, object]] = []
    for i, name in enumerate(names):
        row: Dict[str, object] = {'factor': name}
        for j, other in enumerate(names):
            v = matrix[i, j]
            row[other] = None if np.isnan(v) else float(v)
        rows.append(row)
    return rows
