"""Tests for research.factor_report: the shared report shape CLI and web
both write and read.
"""

import json
import os

import numpy as np
import pytest

from factors.base import Factor
from research.factor_report import build_corr_report, build_factor_panel, build_report, save_report

from tests.conftest import build_panel

SYMBOLS = ('SA', 'CF', 'AG', 'C', 'JM')
N_BARS = 400


def _market(n_bars=N_BARS, symbols=SYMBOLS, seed=0):
    rng = np.random.default_rng(seed)
    panels = {}
    for offset, sym in enumerate(symbols):
        t = np.arange(n_bars, dtype='float64')
        close = 100.0 + 0.04 * t + 3.0 * np.sin(t / 14.0 + offset) + rng.normal(0, 0.3, n_bars)
        close = np.abs(close) + 10.0
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) + 0.4
        low = np.minimum(open_, close) - 0.4
        oi = rng.uniform(5000, 6000, n_bars)
        volume = rng.uniform(1000, 2000, n_bars)
        weighted = {
            'open': open_, 'high': high, 'low': low, 'close': close,
            'settle': close.copy(), 'oi': oi, 'volume': volume, 'session': np.ones(n_bars),
        }
        code = f'{sym}C1'
        contracts = {code: {i: (open_[i], high[i], low[i], close[i], close[i], oi[i], volume[i])
                             for i in range(n_bars)}}
        panels[sym] = build_panel(sym, n_bars, weighted=weighted, contracts=contracts,
                                   contract_by_bar=[code] * n_bars, first_bar=0)

    from core.market import MarketData
    dates = np.array(
        [np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(n_bars)],
        dtype='datetime64[D]',
    )
    return MarketData(dates=dates, products=panels)


class _Momentum(Factor):
    params = {'lookback': 20}

    def compute_symbol(self, ctx, sym):
        from factors.primitives import ts_return
        return ts_return(ctx.close(sym), self.p['lookback'], 0)


class _Volatility(Factor):
    direction = -1

    def compute_symbol(self, ctx, sym):
        from factors.primitives import ts_std
        close = ctx.close(sym)
        log_ret = np.empty_like(close)
        log_ret[0] = np.nan
        log_ret[1:] = np.diff(np.log(close))
        return ts_std(log_ret, 20)


def test_build_report_has_every_section():
    market = _market()
    report = build_report(market, _Momentum(), horizons=[1, 5, 10], n_groups=3)
    for key in (
        'factor', 'params', 'symbols', 'return_source', 'n_groups', 'horizons',
        'coverage', 'ic_decay', 'annual_slices', 'quantiles', 'turnover',
        'autocorr', 'ic_curve', 'quantile_curve',
    ):
        assert key in report
    assert set(report['ic_decay']) == {'1', '5', '10'}
    assert report['quantiles']['horizon'] == 1
    assert report['factor'] == '_Momentum'
    assert report['params'] == {'lookback': 20}


def test_build_report_curves_are_json_safe_and_aligned():
    market = _market()
    report = build_report(market, _Momentum(), horizons=[1, 5], n_groups=3)

    ic_curve = report['ic_curve']
    assert len(ic_curve['dates']) == market.n_bars
    assert len(ic_curve['cumulative_ic']) == market.n_bars
    assert all(isinstance(v, float) for v in ic_curve['cumulative_ic'])
    assert not any(v != v for v in ic_curve['cumulative_ic'])  # no NaN leaked into the curve

    qc = report['quantile_curve']
    assert len(qc['dates']) == len(qc['curves'])
    assert len(qc['curves'][0]) == report['n_groups']


def test_build_report_annual_slices_use_string_year_keys():
    market = _market(n_bars=800)  # spans into a second year
    report = build_report(market, _Momentum(), horizons=[1])
    assert all(isinstance(y, str) for y in report['annual_slices'])
    assert '2020' in report['annual_slices']


def test_build_report_respects_return_source():
    market = _market()
    weighted = build_report(market, _Momentum(), horizons=[1], return_source='weighted')
    exec_ = build_report(market, _Momentum(), horizons=[1], return_source='exec')
    assert weighted['return_source'] == 'weighted'
    assert exec_['return_source'] == 'exec'


def test_build_factor_panel_matches_manual_construction():
    from factors.base import FactorContext, compute_factor
    market = _market()
    factor = _Momentum()
    via_helper = build_factor_panel(factor, market)
    ctx = FactorContext(market, market.symbols)
    manual = compute_factor(factor, ctx)
    np.testing.assert_array_equal(via_helper.values, manual.values)


def test_build_corr_report_returns_json_friendly_rows():
    market = _market()
    panels = {
        'momentum': build_factor_panel(_Momentum(), market),
        'volatility': build_factor_panel(_Volatility(), market),
    }
    result = build_corr_report(market, panels, horizon=1)
    assert result['horizon'] == 1
    assert result['symbols'] == list(market.symbols)

    cs_rows = result['correlation_matrix']
    names = {r['factor'] for r in cs_rows}
    assert names == {'momentum', 'volatility'}
    for row in cs_rows:
        assert row[row['factor']] == pytest.approx(1.0)

    ic_rows = result['ic_correlation']
    assert {r['factor'] for r in ic_rows} == {'momentum', 'volatility'}


# ---------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------

def test_save_report_writes_a_timestamped_json_file(tmp_path):
    market = _market()
    report = build_report(market, _Momentum(), horizons=[1])
    path = save_report(report, 'momentum_report', str(tmp_path))
    assert os.path.dirname(path) == str(tmp_path)
    name = os.path.basename(path)
    assert name.startswith('momentum_report_') and name.endswith('.json')

    with open(path, encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded['factor'] == '_Momentum'
    assert loaded['ic_curve']['dates'][0] == '2020-01-01'


def test_save_report_creates_the_results_directory(tmp_path):
    nested = os.path.join(str(tmp_path), 'a', 'b')
    path = save_report({'x': 1}, 'x', nested)
    assert os.path.isfile(path)


def test_save_report_handles_nan_and_numpy_scalars(tmp_path):
    payload = {'a': float('nan'), 'b': np.float64(1.5), 'c': np.int64(3)}
    path = save_report(payload, 'weird', str(tmp_path))
    with open(path, encoding='utf-8') as f:
        loaded = json.load(f)
    assert loaded['a'] is None
    assert loaded['b'] == 1.5
    assert loaded['c'] == 3
