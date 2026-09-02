"""Live signal generation: what to trade at the next session's open.

Strategy-agnostic and model-agnostic. ``live.signal`` replays loaded history
to its last bar and reads the order the engine never got to fill; it works
the same for a plain strategy out of ``strategies/`` and for one gated by a
``meta_runner.py fit`` model, because the only difference between the two is
which class the engine was handed.

Driven by ``live_runner.py``; ``meta_runner.py signal`` is a thin alias for
the model-backed half.
"""

from __future__ import annotations

from live.report import emit, render, save
from live.signal import (
    SignalReport,
    SignalRow,
    SignalSpec,
    compute_signal,
    spec_from_model,
)

__all__ = [
    'SignalSpec', 'SignalRow', 'SignalReport', 'compute_signal', 'spec_from_model',
    'render', 'save', 'emit',
]
