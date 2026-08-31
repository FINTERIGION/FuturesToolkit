"""Registered CZCE products and symbol helpers.

Commission is per product, one of:
  commission_rate      fraction of notional (price × lots × multiplier)
  commission_per_lot   fixed CNY per lot
If both are set, ``commission_per_lot`` wins.

Rolling is per product too:
  main_months          delivery months that carry the liquidity, e.g. (1, 5, 9)
  roll_lead_months     how far ahead of delivery to roll out (default 1)
A contract is rolled out of on the first calendar day of the month
``roll_lead_months`` before its delivery month -- the 05 contract is dropped
on April 1st. Both keys are optional; omitting them uses the CZCE defaults
below, so a new product only needs an entry when its main months differ.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

_CONTRACT_RE = re.compile(r'^([A-Za-z]+)(\d+)$')
_CONTRACT_DECADE_RE = re.compile(r'^([A-Za-z]+)(\d+)_(\d{6})$')
_WEIGHTED_SUFFIX = '_weighted'

# CZCE chemicals/softs concentrate liquidity in the 01/05/09 contracts, and
# the standing convention is to leave a contract one month before delivery.
DEFAULT_MAIN_MONTHS = (1, 5, 9)
DEFAULT_ROLL_LEAD_MONTHS = 1

PRODUCTS: Dict[str, dict] = {
    'SA': {
        'exchange': 'CZCE',
        'name': 'soda_ash',
        'name_zh': '纯碱',
        'start_year': 2019,
        'main_months': (1, 5, 9),
        'multiplier': 20,
        'margin_rate': 0.12,
        'commission_rate': 0.0001,
    },
    'FG': {
        'exchange': 'CZCE',
        'name': 'glass',
        'name_zh': '玻璃',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 20,
        'margin_rate': 0.13,
        'commission_per_lot': 2.0,
    },
    'CF': {
        'exchange': 'CZCE',
        'name': 'cotton',
        'name_zh': '棉花',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 5,
        'margin_rate': 0.11,
        'commission_per_lot': 4.3,
    },
    'MA': {
        'exchange': 'CZCE',
        'name': 'methanol',
        'name_zh': '甲醇',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 10,
        'margin_rate': 0.11,
        'commission_rate': 0.0001,
    },
    'TA': {
        'exchange': 'CZCE',
        'name': 'pta',
        'name_zh': 'PTA',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 5,
        'margin_rate': 0.11,
        'commission_per_lot': 3.0,
    },
    'SR': {
        'exchange': 'CZCE',
        'name': 'white_sugar',
        'name_zh': '白糖',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 10,
        'margin_rate': 0.10,
        'commission_per_lot': 3.0,
    },
    'OI': {
        'exchange': 'CZCE',
        'name': 'rapeseed_oil',
        'name_zh': '菜油',
        'start_year': 2015,
        'main_months': (1, 5, 9),
        'multiplier': 10,
        'margin_rate': 0.11,
        'commission_per_lot': 2.0,
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
    """Extract a product code from a feed name or contract code.

    ``SA2505`` -> ``SA``, ``FG_weighted`` -> ``FG``, ``CF`` -> ``CF``.
    Decade-disambiguated feeds such as ``FG609_201609`` -> ``FG``.
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
        return prefix or None
    decade = _CONTRACT_DECADE_RE.match(text)
    if decade:
        return decade.group(1).upper()
    match = _CONTRACT_RE.match(text)
    if match:
        return match.group(1).upper()
    return None
