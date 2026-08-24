"""Registered CZCE products and symbol helpers.

Commission is per product, one of:
  commission_rate      fraction of notional (price × lots × multiplier)
  commission_per_lot   fixed CNY per lot
If both are set, ``commission_per_lot`` wins.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

_CONTRACT_RE = re.compile(r'^([A-Za-z]+)(\d+)$')
_CONTRACT_DECADE_RE = re.compile(r'^([A-Za-z]+)(\d+)_(\d{6})$')
_WEIGHTED_SUFFIX = '_weighted'

PRODUCTS: Dict[str, dict] = {
    'SA': {
        'exchange': 'CZCE',
        'name': 'soda_ash',
        'name_zh': '纯碱',
        'start_year': 2019,
        'multiplier': 20,
        'margin_rate': 0.12,
        'commission_rate': 0.0008,
    },
    'FG': {
        'exchange': 'CZCE',
        'name': 'glass',
        'name_zh': '玻璃',
        'start_year': 2015,
        'multiplier': 20,
        'margin_rate': 0.13,
        'commission_per_lot': 24.0,
    },
    'CF': {
        'exchange': 'CZCE',
        'name': 'cotton',
        'name_zh': '棉花',
        'start_year': 2015,
        'multiplier': 5,
        'margin_rate': 0.11,
        'commission_per_lot': 17.2,
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
