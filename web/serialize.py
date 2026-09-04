"""JSON-safe conversion for engine output.

The core engine was never written with JSON in mind: ``compute_metrics``
returns literal ``float('inf')`` for a few ratios
(``core/metrics.py``'s ``calmar_ratio``, ``profit_factor``,
``profit_loss_ratio``), ``equity_records``/``trade_logs`` carry
``datetime.date`` objects, and anything that touched numpy carries numpy
scalar types. Python's own ``json`` module happily emits bare ``Infinity``/
``NaN`` tokens for the first case -- which is not valid JSON and throws in
every browser's ``JSON.parse`` -- and raises ``TypeError`` outright for the
other two. One recursive walk here handles all three, once, so no router
has to know which fields might be affected.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np


def jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into something ``json.dumps`` accepts.

    - ``float('inf')`` / ``-inf`` / ``NaN`` -> ``None`` (the frontend renders
      ``None`` as "∞"/"n/a" once it knows the field is one of these; see
      ``INF_FIELDS`` below for which ones).
    - ``datetime.date`` / ``datetime.datetime`` -> ISO 8601 string.
    - numpy scalars / arrays -> native Python via ``.item()`` / ``.tolist()``.
    - dict / list / tuple -> walked recursively.
    - everything else is returned unchanged (str, int, bool, None, and any
      plain float that is already finite).
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return str(obj)
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj


# Fields in ``compute_metrics``'s output (core/metrics.py) that can be
# +inf when a ratio's denominator is a run with no losing trades at all --
# a strategy that never lost is not a data error, so the field stays present
# with its value nulled by `jsonable`, and the frontend renders it as "∞"
# rather than blank by checking membership here.
INF_CAPABLE_METRIC_FIELDS = ('calmar_ratio', 'profit_factor', 'profit_loss_ratio')


def annotate_inf_metrics(metrics: dict) -> dict:
    """Return ``jsonable(metrics)`` plus an explicit ``<field>_is_inf`` flag
    for every field in ``INF_CAPABLE_METRIC_FIELDS`` that was infinite.

    Doing this once here, rather than asking the frontend to special-case
    "this metric happened to come back null", is what lets the UI render
    ∞ vs n/a vs 0 correctly without knowing anything about how the ratio is
    computed.
    """
    out = jsonable(metrics)
    if not isinstance(out, dict):
        return out
    for field in INF_CAPABLE_METRIC_FIELDS:
        raw = metrics.get(field)
        out[f'{field}_is_inf'] = isinstance(raw, float) and math.isinf(raw) and raw > 0
    return out
