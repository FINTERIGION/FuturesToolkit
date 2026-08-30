"""TA-Lib input guard.

TA-Lib silently returns an all-NaN array when a NaN sits *inside* the
series (not just as a leading run before the first valid value):

    >>> import talib, numpy as np
    >>> talib.SMA(np.array([10, 11, np.nan, 13, 14, 15, 16, 17, 18, 19]), 3)
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])

Leading NaN is fine (TA-Lib skips it correctly); embedded NaN is a silent
footgun. ``guard`` turns it into a loud, actionable error instead.
"""

from __future__ import annotations

import numpy as np


def guard(array, name: str = 'series') -> np.ndarray:
    """Raise if ``array`` has NaN after its first valid value; else return it as float64."""
    arr = np.asarray(array, dtype='float64')
    isnan = np.isnan(arr)
    if not isnan.any():
        return np.ascontiguousarray(arr)
    first = int(np.argmax(~isnan)) if not isnan.all() else len(arr)
    if isnan[first:].any():
        bad = first + np.flatnonzero(isnan[first:])[:5]
        raise ValueError(
            f"{name} has NaN after its first valid value (bar {bad.tolist()}); "
            f"TA-Lib would silently return an all-NaN array for this input. "
            f"Check data alignment."
        )
    return np.ascontiguousarray(arr)
