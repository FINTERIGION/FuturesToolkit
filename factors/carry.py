"""Term structure / carry: backwardation (the near contract priced above the
far one) signals spot tightness and pays a roll return for holding the near
month (see docs/factors.md, "期限结构 / Carry"). ``direction = 1``: steeper
backwardation scores higher.

Needs more than the OI-weighted continuous series can offer -- carry is a
statement about two *simultaneously live* contracts, which is exactly what
``FactorContext.contracts`` exposes and the weighted series (one blended
price per bar) cannot represent. The data is already on disk and already
loaded: ``core.market.ProductPanel.contracts`` keeps every contract's real
prints, not just the calendar-active one.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.params import Int
from datafeed.roll_calendar import parse_contract_expiry

from .base import Factor

Expiry = Tuple[int, int]


def _contract_expiry(code: str, asof) -> Optional[Expiry]:
    """Expiry for a contract feed name, honoring the disambiguating
    ``_YYYYMM`` suffix ``datafeed.roll_calendar.contract_feed_name`` appends
    to a CZCE 3-digit code that collides within the loaded window (the
    3-digit short form wraps every 10 years: ``SA509`` is both 2019-09 and
    2029-09).

    The suffix already spells the expiry out exactly, so it is read directly
    rather than re-parsed through the CZCE short form's decade-guessing path
    (``parse_contract_expiry``) -- re-parsing a code precisely *because* it
    was ambiguous would reintroduce the ambiguity the suffix exists to
    resolve. A plain 4-digit code (SHFE/DCE, or CZCE outside a collision)
    falls straight through to ``parse_contract_expiry``.
    """
    if '_' in code:
        _base, _, suffix = code.rpartition('_')
        if len(suffix) == 6 and suffix.isdigit():
            return int(suffix[:4]), int(suffix[4:6])
    return parse_contract_expiry(code, asof)


def _bar_curve(contracts: dict, dates, i: int, min_oi: float) -> list:
    """``[(expiry, settle), ...]`` sorted by expiry, for contracts with a
    real, liquid print at bar ``i``.

    ``min_oi`` screens out the near-worthless quotes an expiring contract can
    still print in its final sessions (``datafeed.data_update`` notes
    expiring contracts printing ``open=high=low=0``) -- a contract with open
    interest above the floor is one someone could actually still have traded,
    which is the bar carry should be measured against.
    """
    asof = dates[i]
    curve = []
    for code, series in contracts.items():
        row = series.row_at(i)
        if row is None:
            continue
        settle, oi = float(row[4]), float(row[5])
        if not (settle > 0 and oi > min_oi):
            continue
        expiry = _contract_expiry(code, asof)
        if expiry is None:
            continue
        curve.append((expiry, settle))
    curve.sort(key=lambda pair: pair[0])
    return curve


class CarryFactor(Factor):
    """Annualized log price slope between the two nearest liquid contracts::

        ln(near_settle / far_settle) / (far_month - near_month) * 12

    Positive (backwardation: near priced above far) is the classical carry
    signal -- storage-cost theory reads it as spot tightness, paid to whoever
    holds the near month through the roll.
    """

    params = {'min_oi': 0}
    space = {'min_oi': Int(0, 5000, step=100)}
    direction = 1

    def compute_symbol(self, ctx, sym):
        contracts = ctx.contracts(sym)
        dates = ctx.dates
        n_bars = len(dates)
        out = np.full(n_bars, np.nan, dtype='float64')
        if not contracts:
            return out
        min_oi = self.p['min_oi']
        for i in range(n_bars):
            curve = _bar_curve(contracts, dates, i, min_oi)
            if len(curve) < 2:
                continue
            (near_exp, near_px), (far_exp, far_px) = curve[0], curve[1]
            month_gap = (far_exp[0] * 12 + far_exp[1]) - (near_exp[0] * 12 + near_exp[1])
            if month_gap <= 0 or near_px <= 0 or far_px <= 0:
                continue
            out[i] = np.log(near_px / far_px) / month_gap * 12.0
        return out
