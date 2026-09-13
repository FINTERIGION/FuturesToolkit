"""Registered products (CZCE, SHFE, DCE) and symbol helpers.

The registry itself lives in ``datafeed/products.json`` (git-tracked; edited
either by hand or through the web panel's Products page) -- this module is
the loader plus the field documentation and every helper that reads it.

``exchange`` selects the download adapter in ``datafeed.sources``; every other
key here is venue-agnostic.

``tick_size`` is the product's minimum price increment, in the same units the
price series is quoted in. It is what makes a slippage setting portable: the
engine charges ``slippage × tick_size`` per fill, so ``--slippage 1`` is one
tick everywhere rather than one price point -- which would be a rounding error
on gold (0.02) and a 10-point gap on copper.

Commission is per product, one of:
  commission_rate      fraction of notional (price × lots × multiplier)
  commission_per_lot   fixed CNY per lot
Exactly one must be set; ``validate_product`` enforces this on every write.

Rolling is per product too:
  main_months          delivery months that carry the liquidity, e.g. (1, 5, 9)
  roll_lead_months     how far ahead of delivery to roll out (default 1)
A contract is rolled out of on the first calendar day of the month
``roll_lead_months`` before its delivery month -- the 05 contract is dropped
on April 1st. Both keys are optional; omitting them uses the module defaults
below, so a product only needs to declare them when its cycle differs -- as
SHFE rebar (01/05/10), gold and silver (06/12), and the non-ferrous metals
(every month) do.

Contract codes are normalised to UPPERCASE + 4-digit YYMM regardless of venue
(``RB2610``, ``C2601``), which is what ``parse_product`` and
``roll_calendar.parse_contract_expiry`` expect.

This registry is a *catalog* of what the toolkit can fetch and cost correctly;
the universe actually traded is each runner's ``DEFAULT_SYMBOLS``, a subset --
most of what is registered here sits out of the traded set. Doing so costs
nothing: SHFE and DCE payloads are per-day and shared, so an extra product on
those venues is a cache hit, which keeps its data fresh and makes switching it
into the traded set a one-word change.

The numbers below are a starting point, not a live feed: exchanges revise
multipliers, margin floors, and fee schedules by notice, and a broker's margin
sits above the exchange floor by an amount only your own account statement
knows. Calibrate them yourself before trusting a backtest's cost model --
``validate_product`` checks only that each entry is well formed, never what
it says.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

_CONTRACT_RE = re.compile(r'^([A-Za-z]+)(\d+)$')
_CONTRACT_DECADE_RE = re.compile(r'^([A-Za-z]+)(\d+)_(\d{6})$')
_WEIGHTED_SUFFIX = '_weighted'
_CODE_RE = re.compile(r'^[A-Z0-9]+$')

DEFAULT_MAIN_MONTHS = (1, 5, 9)
DEFAULT_ROLL_LEAD_MONTHS = 1
SUPPORTED_EXCHANGES = ('CZCE', 'SHFE', 'DCE')

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'products.json')


def _load_registry(path: str = None) -> Dict[str, dict]:
    """Read the registry file, coercing JSON arrays back to the tuples this
    module's helpers expect (``main_months``). Key order is preserved --
    ``json.load`` keeps the file's order, and ``list_products`` depends on it.
    """
    with open(path or REGISTRY_PATH, encoding='utf-8') as f:
        raw = json.load(f)
    out: Dict[str, dict] = {}
    for code, meta in raw.items():
        entry = dict(meta)
        if 'main_months' in entry:
            entry['main_months'] = tuple(entry['main_months'])
        out[code] = entry
    return out


PRODUCTS: Dict[str, dict] = _load_registry()


def validate_product(code: str, meta: dict) -> dict:
    """Validate one registry entry's shape and return it normalized.

    Codifies the invariants this repo's own test suite checks against every
    entry (``tests/test_products.py``), so a product added or edited through
    the web panel is held to the same bar as one hand-written here. Raises
    ``ValueError`` naming the first offending field -- callers (the web API)
    turn that into a field-level 422.
    """
    norm_code = normalize_symbol(code)
    if not norm_code or not _CODE_RE.match(norm_code):
        raise ValueError(f'Invalid product code {code!r}: use letters/digits only')

    out = dict(meta)

    exchange = out.get('exchange')
    if exchange not in SUPPORTED_EXCHANGES:
        raise ValueError(f'exchange must be one of {SUPPORTED_EXCHANGES}, got {exchange!r}')

    start_year = out.get('start_year')
    if not isinstance(start_year, int) or isinstance(start_year, bool) or not (1990 < start_year <= 2100):
        raise ValueError(f'start_year must be an int in (1990, 2100], got {start_year!r}')

    name = out.get('name')
    if not name or not isinstance(name, str):
        raise ValueError('name must be a non-empty string')
    name_zh = out.get('name_zh')
    if not name_zh or not isinstance(name_zh, str):
        raise ValueError('name_zh must be a non-empty string')

    multiplier = out.get('multiplier')
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or multiplier <= 0:
        raise ValueError(f'multiplier must be > 0, got {multiplier!r}')

    tick = out.get('tick_size')
    if not isinstance(tick, (int, float)) or isinstance(tick, bool) or not (0 < tick <= 100):
        raise ValueError(f'tick_size must be in (0, 100], got {tick!r}')

    margin_rate = out.get('margin_rate', 0.10)
    if not isinstance(margin_rate, (int, float)) or isinstance(margin_rate, bool) or not (0 < margin_rate < 1):
        raise ValueError(f'margin_rate must be in (0, 1), got {margin_rate!r}')
    out['margin_rate'] = float(margin_rate)

    has_rate = out.get('commission_rate') is not None
    has_per_lot = out.get('commission_per_lot') is not None
    if has_rate == has_per_lot:
        raise ValueError(
            'exactly one of commission_rate or commission_per_lot must be set '
            f'(got commission_rate={out.get("commission_rate")!r}, '
            f'commission_per_lot={out.get("commission_per_lot")!r})'
        )
    if has_rate:
        out['commission_rate'] = float(out['commission_rate'])
        out.pop('commission_per_lot', None)
    else:
        out['commission_per_lot'] = float(out['commission_per_lot'])
        out.pop('commission_rate', None)

    if 'main_months' in out:
        out['main_months'] = normalize_main_months(out['main_months'])
    if 'roll_lead_months' in out:
        lead = out['roll_lead_months']
        if not isinstance(lead, int) or isinstance(lead, bool) or lead < 1:
            raise ValueError(f'roll_lead_months must be an int >= 1, got {lead!r}')

    out['multiplier'] = float(multiplier)
    out['tick_size'] = float(tick)
    return out


def save_registry(registry: Dict[str, dict], path: str = None) -> None:
    """Validate every entry, then persist atomically and reload in place.

    Atomic (tmp file + ``os.replace``) so a validation failure or a crash
    mid-write never corrupts the file on disk -- either every entry passes
    and the whole file is replaced in one step, or nothing is written.
    ``PRODUCTS`` is updated in place afterwards, so every module that already
    holds ``from datafeed.products import PRODUCTS`` sees the change without
    re-importing.
    """
    target = path or REGISTRY_PATH
    normalized = {
        normalize_symbol(code): validate_product(code, meta)
        for code, meta in registry.items()
    }

    directory = os.path.dirname(target) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix='.products.', suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    reload_registry(target)


def reload_registry(path: str = None) -> None:
    """Re-read the registry file and update ``PRODUCTS`` in place."""
    fresh = _load_registry(path or REGISTRY_PATH)
    PRODUCTS.clear()
    PRODUCTS.update(fresh)


def list_products() -> List[str]:
    """Return registered product codes in definition order."""
    return list(PRODUCTS)


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def get_product(symbol: str) -> dict:
    key = normalize_symbol(symbol)
    if key not in PRODUCTS:
        supported = ', '.join(list_products())
        raise KeyError(f'Unknown symbol {symbol!r}; supported: {supported}')
    return PRODUCTS[key]


def product_costs(symbol: str) -> dict:
    """Return multiplier, margin, and commission settings for a product."""
    meta = get_product(symbol)
    per_lot = meta.get('commission_per_lot')
    rate = meta.get('commission_rate')
    if per_lot is not None:
        mode = 'per_lot'
        per_lot = float(per_lot)
        rate = None
    else:
        mode = 'rate'
        per_lot = None
        rate = float(rate if rate is not None else 0.0002)
    return {
        'multiplier': meta['multiplier'],
        'margin_rate': float(meta.get('margin_rate', 0.10)),
        'commission_mode': mode,
        'commission_rate': rate,
        'commission_per_lot': per_lot,
    }


def tick_size(symbol: str) -> float:
    """Return the product's minimum price increment.

    Slippage is quoted in ticks, so this is the multiplier the engine applies
    to its ``slippage`` setting -- see ``Engine._slipped``.
    """
    return float(get_product(symbol)['tick_size'])


def normalize_main_months(months) -> Tuple[int, ...]:
    """Validate a delivery-month list and return it sorted and de-duplicated."""
    out = sorted({int(m) for m in months})
    if not out:
        raise ValueError('main_months must not be empty')
    bad = [m for m in out if not 1 <= m <= 12]
    if bad:
        raise ValueError(f'main_months must be in 1..12, got {bad}')
    return tuple(out)


def roll_rule(symbol: str) -> dict:
    """Return the roll calendar settings for a product.

    ``main_months`` are the delivery months actually traded; ``lead_months``
    is how many months before delivery the roll happens, so a product on
    (1, 5, 9) with a one-month lead leaves the 05 contract on April 1st.
    """
    meta = get_product(symbol)
    return {
        'main_months': normalize_main_months(
            meta.get('main_months', DEFAULT_MAIN_MONTHS)
        ),
        'lead_months': int(meta.get('roll_lead_months', DEFAULT_ROLL_LEAD_MONTHS)),
    }


def require_products(symbols: Iterable[str]) -> List[str]:
    """Validate and de-duplicate a product list, preserving order."""
    out: List[str] = []
    for raw in symbols:
        key = normalize_symbol(raw)
        get_product(key)
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError('No products specified')
    return out


def weighted_feed_name(symbol: str) -> str:
    return f'{normalize_symbol(symbol)}_weighted'


def parse_product(code) -> Optional[str]:
    """Extract a **registered** product code from a feed name or contract code.

    ``SA2505`` -> ``SA``, ``CF_weighted`` -> ``CF``, ``CF`` -> ``CF``.
    Decade-disambiguated feeds such as ``SA609_201609`` -> ``SA``.

    ``None`` for anything this registry does not carry, including a string
    that is shaped like a contract code but names nothing here (``XX2505``).
    Returning the letter run regardless, as this used to, produced a plausible
    code that every costing and rolling helper then rejected -- so the failure
    surfaced as a ``KeyError`` inside ``product_costs`` somewhere downstream
    rather than at the point the unknown string came in. Callers already read
    ``None`` as "not a product"; this just makes it mean that consistently.
    """
    if code is None:
        return None
    text = str(code).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in PRODUCTS:
        return upper
    if upper.endswith('_WEIGHTED'):
        prefix = upper[: -len('_WEIGHTED')]
        return prefix if prefix in PRODUCTS else None
    decade = _CONTRACT_DECADE_RE.match(text)
    if decade:
        prefix = decade.group(1).upper()
        return prefix if prefix in PRODUCTS else None
    match = _CONTRACT_RE.match(text)
    if match:
        prefix = match.group(1).upper()
        return prefix if prefix in PRODUCTS else None
    return None
