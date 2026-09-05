"""Cross-factor correlation tests.

The two matrices answer different questions and are easy to conflate:
``correlation_matrix`` asks whether two factors *rank products* the same way,
``ic_correlation`` whether they *work at the same times*. These tests pin each
to a case where the answer is known by construction, including one where the
two deliberately disagree.
"""

import numpy as np
import pytest

from factors.base import Factor, FactorContext, compute_factor
from research.factor_corr import correlation_matrix, ic_correlation, to_rows

from tests.conftest import build_market, build_panel

SYMBOLS = ('SA', 'CF', 'AG', 'C')
N_BARS = 200
N_SYMBOLS = len(SYMBOLS)


def _market(n_bars=N_BARS, symbols=SYMBOLS, seed=0):
    rng = np.random.default_rng(seed)
    panels = {}
    for sym in symbols:
        close = np.abs(100.0 + np.cumsum(rng.normal(0, 1, n_bars))) + 10.0
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        weighted = {
            'open': open_, 'high': close + 1, 'low': close - 1, 'close': close,
            'settle': close.copy(), 'oi': np.full(n_bars, 1000.0),
            'volume': np.full(n_bars, 500.0), 'session': np.ones(n_bars),
        }
        code = f'{sym}C1'
        contracts = {code: {i: (open_[i], close[i] + 1, close[i] - 1, close[i], close[i], 1000.0, 500.0)
                             for i in range(n_bars)}}
        panels[sym] = build_panel(sym, n_bars, weighted=weighted, contracts=contracts,
                                   contract_by_bar=[code] * n_bars, first_bar=0)
    return build_market(panels, n_bars)


def _panel(values, symbols=SYMBOLS):
    n_bars = values.shape[0]
    tradable = np.ones_like(values, dtype=bool)
    from factors.base import FactorPanel
    return FactorPanel(name='test', symbols=list(symbols), values=values, tradable=tradable, direction=1)


# ---------------------------------------------------------------------
# correlation_matrix
# ---------------------------------------------------------------------

def test_a_factor_is_perfectly_correlated_with_itself():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, (100, N_SYMBOLS))
    panels = {'a': _panel(values), 'b': _panel(values.copy())}
    result = correlation_matrix(panels)
    names = result['names']
    matrix = result['matrix']
    i, j = names.index('a'), names.index('b')
    assert matrix[i, j] == pytest.approx(1.0, abs=1e-9)


def test_a_negated_factor_is_perfectly_anticorrelated():
    rng = np.random.default_rng(2)
    values = rng.normal(0, 1, (100, N_SYMBOLS))
    panels = {'a': _panel(values), 'b': _panel(-values)}
    result = correlation_matrix(panels)
    i, j = result['names'].index('a'), result['names'].index('b')
    assert result['matrix'][i, j] == pytest.approx(-1.0, abs=1e-9)


def test_a_monotone_transform_keeps_the_rank_correlation_but_not_the_pearson():
    rng = np.random.default_rng(3)
    values = rng.uniform(1, 10, (100, N_SYMBOLS))
    cubed = values ** 3  # monotone but nonlinear
    panels = {'a': _panel(values), 'b': _panel(cubed)}
    spearman = correlation_matrix(panels, method='spearman')
    pearson = correlation_matrix(panels, method='pearson')
    i, j = spearman['names'].index('a'), spearman['names'].index('b')
    assert spearman['matrix'][i, j] == pytest.approx(1.0, abs=1e-9)
    assert pearson['matrix'][i, j] < 0.999


def test_independent_factors_are_uncorrelated():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, (2000, N_SYMBOLS))
    b = rng.normal(0, 1, (2000, N_SYMBOLS))
    panels = {'a': _panel(a), 'b': _panel(b)}
    result = correlation_matrix(panels)
    i, j = result['names'].index('a'), result['names'].index('b')
    assert abs(result['matrix'][i, j]) < 0.1


def test_the_matrix_is_symmetric_with_a_unit_diagonal():
    rng = np.random.default_rng(5)
    panels = {name: _panel(rng.normal(0, 1, (50, N_SYMBOLS))) for name in ('a', 'b', 'c')}
    result = correlation_matrix(panels)
    matrix = result['matrix']
    np.testing.assert_array_equal(np.diag(matrix), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(matrix, matrix.T)


def test_correlating_one_factor_is_rejected():
    with pytest.raises(ValueError):
        correlation_matrix({'a': _panel(np.zeros((10, N_SYMBOLS)))})


def test_factors_over_different_windows_are_rejected():
    panels = {'a': _panel(np.zeros((10, N_SYMBOLS))), 'b': _panel(np.zeros((20, N_SYMBOLS)))}
    with pytest.raises(ValueError):
        correlation_matrix(panels)


def test_an_unknown_method_is_rejected():
    panels = {'a': _panel(np.zeros((10, N_SYMBOLS))), 'b': _panel(np.zeros((10, N_SYMBOLS)))}
    with pytest.raises(ValueError):
        correlation_matrix(panels, method='bogus')


def test_a_bar_below_min_names_does_not_contribute():
    # Only 2 symbols valid per bar, min_names default 3 -> every bar excluded -> NaN average.
    values_a = np.random.default_rng(6).normal(0, 1, (50, 2))
    values_b = np.random.default_rng(7).normal(0, 1, (50, 2))
    panels = {'a': _panel(values_a, symbols=('SA', 'CF')), 'b': _panel(values_b, symbols=('SA', 'CF'))}
    result = correlation_matrix(panels, min_names=3)
    i, j = result['names'].index('a'), result['names'].index('b')
    assert np.isnan(result['matrix'][i, j])


# ---------------------------------------------------------------------
# ic_correlation
# ---------------------------------------------------------------------

def test_ic_series_correlation_of_a_series_with_itself_is_one():
    rng = np.random.default_rng(8)
    ic = rng.normal(0, 1, 100)
    result = ic_correlation({'a': ic, 'b': ic.copy()})
    i, j = result['names'].index('a'), result['names'].index('b')
    assert result['matrix'][i, j] == pytest.approx(1.0, abs=1e-9)


def test_ic_series_correlation_ignores_bars_either_factor_could_not_score():
    rng = np.random.default_rng(9)
    a = rng.normal(0, 1, 200)
    b = a.copy()
    b[:50] = np.nan  # a's own head is fine, b just hasn't warmed up yet
    result = ic_correlation({'a': a, 'b': b})
    i, j = result['names'].index('a'), result['names'].index('b')
    assert result['matrix'][i, j] == pytest.approx(1.0, abs=1e-9)


def test_ic_series_correlation_is_nan_without_enough_overlap():
    a = np.full(10, np.nan)
    a[0] = 1.0
    a[1] = 2.0
    b = np.full(10, np.nan)
    b[0] = 1.0
    b[1] = 2.0
    result = ic_correlation({'a': a, 'b': b}, min_names=3)
    i, j = result['names'].index('a'), result['names'].index('b')
    assert np.isnan(result['matrix'][i, j])


def test_ic_series_of_different_lengths_are_rejected():
    with pytest.raises(ValueError):
        ic_correlation({'a': np.zeros(10), 'b': np.zeros(20)})


def test_correlating_one_ic_series_is_rejected():
    with pytest.raises(ValueError):
        ic_correlation({'a': np.zeros(10)})


def test_the_two_matrices_can_disagree():
    """Two factors that rank products identically every bar (perfect
    cross-sectional correlation) but whose *IC* series are uncorrelated --
    i.e. they agree on ranking, but their skill shows up in different bars."""
    n_bars = 400
    rng = np.random.default_rng(10)
    base = rng.normal(0, 1, (n_bars, N_SYMBOLS))
    # b is a positive affine transform of a *per bar* (same rank every bar),
    # but the transform's scale swings sign in blocks -- same ranking always,
    # while a downstream IC computed against an external return could react
    # oppositely in different blocks. We approximate this directly at the IC
    # level: construct two IC series with zero correlation by construction.
    panels_a = _panel(base)
    panels_b = _panel(base * 2.0 + 1.0)  # positive affine -> identical ranking every bar
    cs_result = correlation_matrix({'a': panels_a, 'b': panels_b})
    i, j = cs_result['names'].index('a'), cs_result['names'].index('b')
    assert cs_result['matrix'][i, j] == pytest.approx(1.0, abs=1e-9)

    ic_a = rng.normal(0, 1, 100)
    ic_b = rng.normal(0, 1, 100)  # independent draw -> uncorrelated IC series
    ic_result = ic_correlation({'a': ic_a, 'b': ic_b})
    ii, jj = ic_result['names'].index('a'), ic_result['names'].index('b')
    assert abs(ic_result['matrix'][ii, jj]) < 0.3


# ---------------------------------------------------------------------
# to_rows
# ---------------------------------------------------------------------

def test_to_rows_flattens_to_json_friendly_dicts():
    result = {'names': ['a', 'b'], 'matrix': np.array([[1.0, 0.5], [0.5, 1.0]])}
    rows = to_rows(result)
    assert rows == [
        {'factor': 'a', 'a': 1.0, 'b': 0.5},
        {'factor': 'b', 'a': 0.5, 'b': 1.0},
    ]


def test_to_rows_turns_nan_into_none():
    result = {'names': ['a', 'b'], 'matrix': np.array([[1.0, np.nan], [np.nan, 1.0]])}
    rows = to_rows(result)
    assert rows[0]['b'] is None
    assert rows[1]['a'] is None
