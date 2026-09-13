"""DataManager's contract-level loading, on a throwaway data directory."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datafeed.data_manager import DataManager
from datafeed.products import roll_rule
from datafeed.roll_calendar import target_expiry

_NO_PRICE = {'open': np.nan, 'high': np.nan, 'low': np.nan}


def test_a_contract_row_with_a_blank_open_is_not_a_print(tmp_path, monkeypatch):
    """SHFE lists every contract every day and, on a day one did not trade,
    publishes close and settle with open/high/low left blank. Loaded as a
    print, that row put a NaN open under any fill or roll landing on it, and
    the NaN equity that followed killed every risk-sized AG or AU run.
    """
    monkeypatch.setattr(DataManager, 'DATA_DIR', str(tmp_path))
    dates = pd.bdate_range('2025-03-03', periods=5)
    rule = roll_rule('AG')
    expiries = {target_expiry(d, rule['main_months'], rule['lead_months']) for d in dates}
    assert len(expiries) == 1, 'the window must not straddle a roll'
    year, month = expiries.pop()
    assert month <= 10
    live = f'AG{year % 100:02d}{month:02d}'
    idle = f'AG{year % 100:02d}{month + 2:02d}'   # listed, never traded in the window

    rows = []
    for i, day in enumerate(dates):
        px = 8000.0 + i
        prices = _NO_PRICE if i == 2 else {'open': px, 'high': px + 5, 'low': px - 5}
        rows.append({'date': day, 'contract': live, **prices, 'close': px,
                     'oi': 300, 'volume': 0.0 if i == 2 else 100.0, 'settle': px})
        rows.append({'date': day, 'contract': idle, **_NO_PRICE, 'close': px + 20,
                     'oi': 10, 'volume': 0.0, 'settle': px + 20})
    contracts = pd.DataFrame(rows)
    contracts['date'] = contracts['date'].dt.strftime('%Y-%m-%d')
    contracts.to_csv(tmp_path / 'AG.csv', index=False)
    pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'), 'open': 8000.0, 'high': 8005.0, 'low': 7995.0,
        'close': 8000.0, 'settle': 8000.0, 'oi': 300, 'volume': 100.0,
    }).to_csv(tmp_path / 'AG_weighted.csv', index=False)

    bundle = DataManager(symbols=['AG']).get_universe_bundle()['products']['AG']

    frames = bundle['contract_frames']
    assert idle not in frames
    assert list(frames[live].index) == [dates[0], dates[1], dates[3], dates[4]]
    assert not frames[live][['open', 'high', 'low', 'close']].isna().any().any()
    # The untraded day is dark for the product rather than a day to fill on.
    assert bundle['exec_price_df'].loc[dates[2], 'contract'] == ''
    assert bundle['exec_price_df'].loc[dates[1], 'contract'] == live
