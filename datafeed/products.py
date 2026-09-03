"""Registered products (CZCE, SHFE, DCE) and symbol helpers.

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
If both are set, ``commission_per_lot`` wins.

Rolling is per product too:
  main_months          delivery months that carry the liquidity, e.g. (1, 5, 9)
  roll_lead_months     how far ahead of delivery to roll out (default 1)
A contract is rolled out of on the first calendar day of the month
``roll_lead_months`` before its delivery month -- the 05 contract is dropped
on April 1st. Both keys are optional; omitting them uses the module defaults
below, so a product only needs to declare them when its cycle differs -- as
SHFE rebar (01/05/10), silver (even months), and the non-ferrous metals (every
month) do.

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
knows. Calibrate them yourself before trusting a backtest's cost model -- the
tests here check only that each entry is well formed, never what it says.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

_CONTRACT_RE = re.compile(r'^([A-Za-z]+)(\d+)$')
_CONTRACT_DECADE_RE = re.compile(r'^([A-Za-z]+)(\d+)_(\d{6})$')
_WEIGHTED_SUFFIX = '_weighted'

DEFAULT_MAIN_MONTHS = (1, 5, 9)
DEFAULT_ROLL_LEAD_MONTHS = 1

PRODUCTS: Dict[str, dict] = {
    # -- CZCE ---------------------------------------------------------------
    'SA': {
        'exchange': 'CZCE',
        'name': 'soda_ash',
        'name_zh': '纯碱',
        'start_year': 2019,
        'main_months': (1, 5, 9),
        'multiplier': 20,
        'tick_size': 1.0,
        'margin_rate': 0.11,
        'commission_rate': 0.0001,
    },
    'CF': {
        'exchange': 'CZCE',
        'name': 'cotton',
        'name_zh': '棉花',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 5,
        'tick_size': 5.0,
        'margin_rate': 0.10,
        'commission_per_lot': 4.3,
    },
    'FG': {
        'exchange': 'CZCE',
        'name': 'glass',
        'name_zh': '玻璃',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 20,
        'tick_size': 1.0,
        'margin_rate': 0.12,
        'commission_per_lot': 2.0,
    },
    'SR': {
        'exchange': 'CZCE',
        'name': 'white_sugar',
        'name_zh': '白糖',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 10,
        'tick_size': 1.0,
        'margin_rate': 0.09,
        'commission_per_lot': 2.0,
    },
    # -- SHFE ---------------------------------------------------------------
    'AG': {
        'exchange': 'SHFE',
        'name': 'silver',
        'name_zh': '沪银',
        'start_year': 2015,
        'main_months': (2, 4, 6, 8, 10, 12),
        'multiplier': 15,
        'tick_size': 1.0,
        'margin_rate': 0.25,
        'commission_rate': 0.00001,
    },
    'AU': {
        'exchange': 'SHFE',
        'name': 'gold',
        'name_zh': '沪金',
        'start_year': 2015,
        'main_months': (2, 4, 6, 8, 10, 12),
        'multiplier': 1000,
        'tick_size': 0.02,
        'margin_rate': 0.19,
        'commission_per_lot': 10.0,
    },
    'CU': {
        'exchange': 'SHFE',
        'name': 'copper',
        'name_zh': '沪铜',
        'start_year': 2015,
        'main_months': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        'multiplier': 5,
        'tick_size': 10.0,
        'margin_rate': 0.14,
        'commission_rate': 0.00005,
    },
    'AL': {
        'exchange': 'SHFE',
        'name': 'aluminium',
        'name_zh': '沪铝',
        'start_year': 2015,
        'main_months': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        'multiplier': 5,
        'tick_size': 5.0,
        'margin_rate': 0.14,
        'commission_per_lot': 3.0,
    },
    'ZN': {
        'exchange': 'SHFE',
        'name': 'zinc',
        'name_zh': '沪锌',
        'start_year': 2015,
        'main_months': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        'multiplier': 5,
        'tick_size': 5.0,
        'margin_rate': 0.15,
        'commission_per_lot': 3.0,
    },
    'SN': {
        'exchange': 'SHFE',
        'name': 'tin',
        'name_zh': '沪锡',
        'start_year': 2021,
        'main_months': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        'multiplier': 1,
        'tick_size': 10.0,
        'margin_rate': 0.18,
        'commission_per_lot': 3.0,
    },
    'RB': {
        'exchange': 'SHFE',
        'name': 'rebar',
        'name_zh': '螺纹钢',
        'start_year': 2015,
        'main_months': (1, 5, 10),
        'multiplier': 10,
        'tick_size': 1.0,
        'margin_rate': 0.10,
        'commission_rate': 0.0001,
    },
    'HC': {
        'exchange': 'SHFE',
        'name': 'hot_rolled_coil',
        'name_zh': '热卷',
        'start_year': 2015,
        'main_months': (1, 5, 10),
        'multiplier': 10,
        'tick_size': 1.0,
        'margin_rate': 0.10,
        'commission_rate': 0.0001,
    },
    'RU': {
        'exchange': 'SHFE',
        'name': 'rubber',
        'name_zh': '橡胶',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 10,
        'tick_size': 5.0,
        'margin_rate': 0.12,
        'commission_per_lot': 3.0,
    },
    # -- DCE ----------------------------------------------------------------
    'C': {
        'exchange': 'DCE',
        'name': 'corn',
        'name_zh': '玉米',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 10,
        'tick_size': 1.0,
        'margin_rate': 0.11,
        'commission_per_lot': 1.2,
    },
    'JM': {
        'exchange': 'DCE',
        'name': 'coking_coal',
        'name_zh': '焦煤',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 60,
        'tick_size': 0.5,
        'margin_rate': 0.17,
        'commission_rate': 0.0001,
    },
    'LH': {
        'exchange': 'DCE',
        'name': 'live_hog',
        'name_zh': '生猪',
        'start_year': 2021,
        'main_months': (1, 3, 5, 7, 9, 11),
        'multiplier': 16,
        'tick_size': 5.0,
        'margin_rate': 0.11,
        'commission_rate': 0.0002,
    },
}


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
