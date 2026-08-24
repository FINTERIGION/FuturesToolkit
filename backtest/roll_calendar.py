"""Calendar-spread roll: map trading dates to the executable CZCE contract.

Calendar (calendar months, not delivery months):
  Dec / Jan / Feb / Mar  ->  May contract (Dec uses next year's May)
  Apr / May / Jun / Jul  ->  September contract (same year)
  Aug / Sep / Oct / Nov  ->  next year's January contract

Contract codes are resolved from ``data/{symbol}.csv`` (CZCE 3-digit YMM or
4-digit YYMM suffixes such as SA509 / FG2505), not constructed by hand.
"""

from __future__ import annotations

import re
from typing import Dict, Hashable, Iterable, Optional, Tuple

import pandas as pd

Expiry = Tuple[int, int]  # (year, month)

_CONTRACT_RE = re.compile(r'^[A-Za-z]+(\d+)$')


def normalize_contract_code(code) -> str:
    """Strip whitespace so CZCE codes match Backtrader feed names."""
    return str(code).strip().replace(' ', '')


def target_expiry(dt) -> Expiry:
    """Return the (year, month) of the contract that should be traded on ``dt``."""
    ts = pd.Timestamp(dt)
    year, month = int(ts.year), int(ts.month)
    if month == 12:
        return year + 1, 5
    if month in (1, 2, 3):
        return year, 5
    if month in (4, 5, 6, 7):
        return year, 9
    return year + 1, 1


def parse_contract_expiry(code, asof) -> Optional[Expiry]:
    """Parse a CZCE contract code into (expiry_year, expiry_month).

    4-digit suffixes are YYMM (SA2505 -> 2025-05).
    3-digit suffixes are YMM (SA509 -> year ending in 5, month 09),
    disambiguated with ``asof`` so the expiry is not already past.
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

    CZCE 3-digit codes wrap every 10 years (FG501 is 2015-01 and 2025-01).
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
    """Unique feed name; suffix YYYYMM when the 3-digit code is reused."""
    if code in colliding:
        year, month = expiry
        return f'{code}_{year}{month:02d}'
    return code


def build_date_contract_map(
    trading_index: Iterable,
    contracts_df: pd.DataFrame,
    warn: bool = True,
    listed_from=None,
) -> Dict[Hashable, str]:
    """Map each trading date to the calendar contract feed name.

    ``contracts_df`` must contain ``date`` and ``contract`` columns.
    Expiry is parsed per row so 3-digit codes that wrap every decade
    (FG501 in 2015 vs 2025) resolve independently.

    If the target contract has no row on that date, the previous mapped
    contract is kept and a warning is printed (once per target expiry).
    Dates before ``listed_from`` (product listing) are skipped.
    """
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
    prev_code: Optional[str] = None
    warned_expiries = set()

    for dt in index:
        if listed is not None and dt < listed:
            continue
        expiry = target_expiry(dt)
        candidates = names_by_expiry.get(expiry, [])

        code = None
        for cand in candidates:
            if dt in dates_by_key.get((expiry, cand), ()):
                code = cand
                break
        if code is None and candidates:
            # Listed, but no print that session — still tradable after ffill
            # if it has any history on or before this date.
            for cand in candidates:
                earlier = {d for d in dates_by_key.get((expiry, cand), ()) if d <= dt}
                if earlier:
                    code = cand
                    break

        if code is None:
            if prev_code is not None:
                if warn and expiry not in warned_expiries:
                    print(
                        f"[RollCalendar] {dt.date()}: no contract for "
                        f"{expiry[0]}-{expiry[1]:02d}, keeping {prev_code}"
                    )
                    warned_expiries.add(expiry)
                code = prev_code
            elif warn and expiry not in warned_expiries:
                print(
                    f"[RollCalendar] {dt.date()}: no contract for "
                    f"{expiry[0]}-{expiry[1]:02d} and no previous contract"
                )
                warned_expiries.add(expiry)

        if code is not None:
            mapping[dt] = code
            prev_code = code

    return mapping


def mapping_as_dates(mapping: Dict[Hashable, str]) -> Dict:
    """Convert Timestamp keys to ``datetime.date`` for strategy params."""
    return {pd.Timestamp(k).date(): v for k, v in mapping.items()}
