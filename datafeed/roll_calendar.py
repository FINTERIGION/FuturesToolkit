"""Calendar-spread roll: map trading dates to the executable contract.

Venue-agnostic. The calendar is derived from each product's registered
``main_months`` (see ``datafeed.products.roll_rule``) rather than hard-coded:
a contract is rolled out of on the **first calendar day of the month
``lead_months`` before its delivery month**, so the position always sits in
the nearest main contract that has not yet reached its roll date.

With the registry default -- main months (1, 5, 9), one month of lead -- that
reproduces the usual schedule:
  Dec / Jan / Feb / Mar  ->  May contract (Dec uses next year's May)
  Apr / May / Jun / Jul  ->  September contract (same year)
  Aug / Sep / Oct / Nov  ->  next year's January contract

A product off that cycle just declares its own months; nothing here changes.

Contract codes are resolved from ``data/{symbol}.csv``, not constructed by
hand. Both suffix forms are accepted: 4-digit YYMM, which SHFE and DCE always
use and CZCE uses in its files (RB2601 / C2601 / FG2505), and the 3-digit YMM
short form CZCE prints on its own site (SA509).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Hashable, Iterable, Optional, Tuple

import pandas as pd

from .products import (
    DEFAULT_MAIN_MONTHS,
    DEFAULT_ROLL_LEAD_MONTHS,
    normalize_main_months,
)

logger = logging.getLogger(__name__)

Expiry = Tuple[int, int]  # (year, month)

_CONTRACT_RE = re.compile(r'^[A-Za-z]+(\d+)$')


def normalize_contract_code(code) -> str:
    """Strip whitespace so exchange codes match feed names."""
    return str(code).strip().replace(' ', '')


def roll_date(expiry: Expiry, lead_months: int = DEFAULT_ROLL_LEAD_MONTHS):
    """First calendar day of the month ``lead_months`` ahead of delivery.

    This is the day the contract is rolled *out of*: with a one-month lead a
    05 contract is dropped on April 1st, not held into the delivery month.
    """
    year, month = int(expiry[0]), int(expiry[1])
    start = pd.Timestamp(year=year, month=month, day=1)
    return start - pd.DateOffset(months=int(lead_months))


def target_expiry(
    dt,
    main_months: Iterable[int] = DEFAULT_MAIN_MONTHS,
    lead_months: int = DEFAULT_ROLL_LEAD_MONTHS,
) -> Expiry:
    """Return the (year, month) of the contract that should be traded on ``dt``.

    The nearest main-month delivery whose roll date is still ahead of ``dt``.
    Roll dates rise with the delivery month, so the first candidate that
    clears ``dt`` is the closest one.
    """
    ts = pd.Timestamp(dt).normalize()
    months = normalize_main_months(main_months)
    lead = int(lead_months)
    # A lead longer than a year pulls the roll date back whole years, so the
    # scan has to start that far in the past to still find the nearest one.
    first_year = int(ts.year) - lead // 12 - 1
    for year in range(first_year, int(ts.year) + 3):
        for month in months:
            if roll_date((year, month), lead) > ts:
                return year, month
    raise ValueError(
        f'No contract found for {ts.date()} with main_months={months} '
        f'and lead_months={lead}'
    )


def parse_contract_expiry(code, asof) -> Optional[Expiry]:
    """Parse a contract code into (expiry_year, expiry_month).

    4-digit suffixes are YYMM (SA2505, RB2601, C2601 -> 2025-05 / 2026-01).
    3-digit suffixes are the CZCE short form, YMM (SA509 -> year ending in 5,
    month 09), disambiguated with ``asof`` so the expiry is not already past.
    """
    code = normalize_contract_code(code)
    match = _CONTRACT_RE.match(code)
    if not match:
        return None

    digits = match.group(1)
    asof = pd.Timestamp(asof).normalize()

    if len(digits) >= 4:
        yy = int(digits[-4:-2])
        month = int(digits[-2:])
        if not 1 <= month <= 12:
            return None
        return 2000 + yy, month

    if len(digits) == 3:
        year_digit = int(digits[0])
        month = int(digits[1:])
        if not 1 <= month <= 12:
            return None
        best: Optional[pd.Timestamp] = None
        best_exp: Optional[Expiry] = None
        for year in range(int(asof.year) - 2, int(asof.year) + 12):
            if year % 10 != year_digit:
                continue
            exp = pd.Timestamp(year=year, month=month, day=1)
            month_end = exp + pd.offsets.MonthEnd(0)
            if month_end < asof:
                continue
            if best is None or exp < best:
                best = exp
                best_exp = (year, month)
        return best_exp

    return None


def infer_code_expiry(code: str, sample_dates: Iterable) -> Optional[Expiry]:
    """Infer expiry for a contract from the dates it actually traded."""
    dates = list(sample_dates)
    if not dates:
        return None
    first = min(pd.Timestamp(d) for d in dates)
    return parse_contract_expiry(code, first)


def annotate_expiries(contracts_df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``expiry`` column parsed with each row's own date as ``asof``.

    The CZCE 3-digit short form wraps every 10 years (FG501 is both 2015-01
    and 2025-01); 4-digit codes do not, so this only bites on CZCE.
    Expiry must be parsed per print date, not from the code's first-ever bar.
    """
    df = contracts_df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df['contract'] = df['contract'].map(normalize_contract_code)
    df = df.dropna(subset=['date', 'contract'])
    df['expiry'] = [
        parse_contract_expiry(code, dt)
        for code, dt in zip(df['contract'], df['date'])
    ]
    return df.dropna(subset=['expiry'])


def colliding_codes(df: pd.DataFrame, start, end) -> set:
    """Codes that map to more than one expiry inside ``[start, end]``."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    window = df[(df['date'] >= start) & (df['date'] <= end)]
    if window.empty or 'expiry' not in window.columns:
        return set()
    counts = window.groupby('contract')['expiry'].nunique()
    return set(counts[counts > 1].index)


def contract_feed_name(code: str, expiry: Expiry, colliding: set) -> str:
    """Unique feed name; suffix YYYYMM when a 3-digit CZCE code is reused."""
    if code in colliding:
        year, month = expiry
        return f'{code}_{year}{month:02d}'
    return code


def build_date_contract_map(
    trading_index: Iterable,
    contracts_df: pd.DataFrame,
    warn: bool = True,
    listed_from=None,
    main_months: Iterable[int] = DEFAULT_MAIN_MONTHS,
    lead_months: int = DEFAULT_ROLL_LEAD_MONTHS,
) -> Dict[Hashable, str]:
    """Map each trading date to the calendar contract feed name.

    ``contracts_df`` must contain ``date`` and ``contract`` columns.
    Expiry is parsed per row so 3-digit codes that wrap every decade
    (FG501 in 2015 vs 2025) resolve independently.

    ``main_months`` / ``lead_months`` come from the product registry
    (``datafeed.products.roll_rule``) and decide which delivery month is
    active on each date.

    If the target contract has no row on that date, the date is left
    unmapped (not tradable). A warning is printed once per target expiry.
    Dates before ``listed_from`` (product listing) are skipped.
    """
    months = normalize_main_months(main_months)
    lead = int(lead_months)
    if contracts_df.empty:
        raise ValueError("contracts_df is empty; cannot build a roll calendar")

    index = pd.DatetimeIndex(pd.to_datetime(list(trading_index))).normalize()
    df = annotate_expiries(contracts_df)
    if df.empty:
        raise ValueError("contracts_df has no parsable contract expiries")

    colliding = colliding_codes(df, index.min(), index.max()) if len(index) else set()
    listed = (
        pd.Timestamp(listed_from).normalize() if listed_from is not None else None
    )

    dates_by_key: Dict[Tuple[Expiry, str], set] = {}
    names_by_expiry: Dict[Expiry, list] = {}
    for expiry, code, dt in zip(df['expiry'], df['contract'], df['date']):
        name = contract_feed_name(code, expiry, colliding)
        dates_by_key.setdefault((expiry, name), set()).add(dt)
        bucket = names_by_expiry.setdefault(expiry, [])
        if name not in bucket:
            bucket.append(name)

    mapping: Dict[Hashable, str] = {}
    warned_expiries = set()

    for dt in index:
        if listed is not None and dt < listed:
            continue
        expiry = target_expiry(dt, months, lead)
        candidates = names_by_expiry.get(expiry, [])

        code = None
        for cand in candidates:
            if dt in dates_by_key.get((expiry, cand), ()):
                code = cand
                break

        if code is None:
            if warn and expiry not in warned_expiries:
                logger.warning(
                    "%s: no print for %s-%02d, skipping session",
                    dt.date(), expiry[0], expiry[1],
                )
                warned_expiries.add(expiry)
            continue

        mapping[dt] = code

    return mapping


def mapping_as_dates(mapping: Dict[Hashable, str]) -> Dict:
    """Convert Timestamp keys to ``datetime.date`` for strategy params."""
    return {pd.Timestamp(k).date(): v for k, v in mapping.items()}
