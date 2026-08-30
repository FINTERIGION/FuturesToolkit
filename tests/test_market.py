import numpy as np

from core.market import ContractSeries, MarketData, ProductPanel


def test_bar_count_and_symbols():
    dates = np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]')
    panel = ProductPanel(
        symbol='SA',
        weighted={'session': np.array([1.0, 1.0, 1.0])},
        contracts={},
        contract_by_bar=np.array(['', '', ''], dtype=object),
        first_bar=0,
    )
    md = MarketData(dates=dates, products={'SA': panel})
    assert md.n_bars == 3
    assert md.symbols == ['SA']


def test_contract_series_row_at_only_returns_live_bars():
    live_bars = np.array([2, 5, 9], dtype='int64')
    ohlcv = np.array([
        [10, 11, 9, 10, 10, 0, 100],
        [12, 13, 11, 12, 12, 0, 200],
        [15, 16, 14, 15, 15, 0, 300],
    ], dtype='float64')
    cs = ContractSeries(code='SA509', start=2, live_bars=live_bars, ohlcv=ohlcv)

    assert cs.row_at(2)[0] == 10
    assert cs.row_at(5)[0] == 12
    assert cs.row_at(9)[0] == 15
    assert cs.row_at(0) is None    # before it ever printed
    assert cs.row_at(3) is None    # dark bar between two real prints
    assert cs.row_at(10) is None   # after its last print


def test_can_trade_requires_listing_session_and_mapped_contract():
    session = np.array([0.0, 1.0, 1.0, 1.0])
    contract_by_bar = np.array(['', '', 'SA509', ''], dtype=object)
    panel = ProductPanel(
        symbol='SA',
        weighted={'session': session},
        contracts={},
        contract_by_bar=contract_by_bar,
        first_bar=1,
    )
    assert panel.can_trade(0) is False   # not listed yet
    assert panel.can_trade(1) is False   # listed + live session, but no contract mapped
    assert panel.can_trade(2) is True    # listed + live session + mapped contract
    assert panel.can_trade(3) is False   # mapped contract has no print today
