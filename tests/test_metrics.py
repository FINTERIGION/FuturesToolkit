"""Metric definitions that have to mean what everyone else means by the same
name.

Checked against values derived here from the definition rather than against
``compute_metrics``' own output, because the failure mode being guarded is a
formula that is self-consistent and still not the ratio it is labelled as.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from core.metrics import _annualization_factor, compute_metrics

_RETURNS = [0.011, -0.019, 0.014, -0.006, 0.008, -0.012, 0.0, 0.021, -0.028, 0.013,
            0.004, -0.009, 0.017, -0.002, -0.015, 0.006, 0.010, -0.021, 0.009, 0.003]


def _records(returns, initial_cash=100_000.0):
    day = datetime.date(2024, 1, 1)
    equity = initial_cash
    out = []
    for i, r in enumerate(returns):
        equity *= 1.0 + r
        out.append({
            'date': day + datetime.timedelta(days=i),
            'equity': equity,
            'daily_return': r,
            'margin_used': 0.0,
            'available': equity,
            'position': {},
        })
    return out


def test_sortino_uses_downside_deviation_not_the_spread_of_losing_days():
    """Downside deviation is the root mean square of the shortfall below the
    target, over *every* observation.

    The implementation used to take ``excess[excess < 0].std()``, which is a
    different quantity twice over: it centres on the mean of the negatives, so
    it measures how much the bad days differ from one another rather than how
    far they fall, and it divides by the count of negatives rather than the
    sample size. Both are self-consistent and neither is Sortino.
    """
    records = _records(_RETURNS)
    metrics = compute_metrics(records, [], 100_000.0)

    tdy = _annualization_factor([r['date'] for r in records])
    excess = np.array(_RETURNS, dtype=float) - 0.03 / tdy
    downside_deviation = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    expected = excess.mean() / downside_deviation * math.sqrt(tdy)

    assert metrics['sortino_ratio'] == pytest.approx(expected, abs=1e-4)

    # And the two really do disagree -- this is not a distinction without a
    # difference that happens to round the same way.
    superseded = excess.mean() / excess[excess < 0].std() * math.sqrt(tdy)
    assert abs(superseded - expected) / abs(expected) > 0.05


def test_sortino_is_zero_when_nothing_ever_fell_below_the_target():
    """No shortfall means no downside deviation to divide by. The ratio is
    undefined there, and 0.0 is what every other guard in this module returns
    for an undefined denominator -- not an infinity the JSON layer then has to
    special-case."""
    flat = [0.02] * 10          # every day beats the risk-free rate
    metrics = compute_metrics(_records(flat), [], 100_000.0)
    assert metrics['sortino_ratio'] == 0.0
