"""Factor primitive operator tests."""

import numpy as np
import pytest

from factors.primitives import (
    cs_demean, cs_rank, cs_zscore, lag, ts_diff, ts_mean, ts_rank, ts_return,
    ts_std, ts_zscore, winsorize,
)
from strategies.cross_sectional_momentum import _momentum


# ---------------------------------------------------------------------
# lag
# ---------------------------------------------------------------------

def test_lag_shifts_into_the_past_with_a_nan_head():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = lag(arr, 2)
    assert np.isnan(out[:2]).all()
    assert np.array_equal(out[2:], [1.0, 2.0, 3.0])


def test_lag_refuses_to_look_forward():
    with pytest.raises(ValueError):
        lag(np.array([1.0, 2.0, 3.0]), -1)


def test_lag_longer_than_the_series_is_all_nan():
    arr = np.array([1.0, 2.0, 3.0])
    out = lag(arr, 10)
    assert np.isnan(out).all()


# ---------------------------------------------------------------------
# ts_return
# ---------------------------------------------------------------------

def test_ts_return_matches_the_strategy_it_was_lifted_from():
    close = 100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, 200))
    close = np.abs(close) + 1.0
    a = ts_return(close, 20, 5)
    b = _momentum(close, 20, 5)
    np.testing.assert_array_equal(a, b)


def test_ts_return_computes_the_right_ratio():
    close = np.array([100.0, 101.0, 102.0, 110.0, 121.0])
    out = ts_return(close, 2, 0)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(102.0 / 100.0 - 1.0)
    assert out[4] == pytest.approx(121.0 / 102.0 - 1.0)


def test_ts_return_skips_recent_bars():
    close = np.array([100.0, 101.0, 102.0, 110.0, 121.0])
    out = ts_return(close, 2, 1)
    # span = lookback + skip = 3; out[3] compares close[0] with close[2] (skip=1)
    assert out[3] == pytest.approx(102.0 / 100.0 - 1.0)


def test_ts_return_gives_nan_for_a_non_positive_base():
    close = np.array([-5.0, 1.0, 2.0, 3.0])
    out = ts_return(close, 1, 0)
    assert np.isnan(out[1])


# ---------------------------------------------------------------------
# ts_diff / ts_mean / ts_std / ts_zscore
# ---------------------------------------------------------------------

def test_ts_diff():
    arr = np.array([1.0, 3.0, 6.0, 10.0])
    out = ts_diff(arr, 1)
    assert np.isnan(out[0])
    np.testing.assert_array_equal(out[1:], [2.0, 3.0, 4.0])


def test_ts_mean_and_std_are_trailing_and_inclusive():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = ts_mean(arr, 3)
    std = ts_std(arr, 3)
    assert np.isnan(mean[:2]).all()
    assert mean[2] == pytest.approx(2.0)   # mean(1,2,3)
    assert mean[4] == pytest.approx(4.0)   # mean(3,4,5)
    assert std[2] == pytest.approx(np.std([1.0, 2.0, 3.0]))


def test_ts_mean_window_longer_than_the_series_is_all_nan():
    arr = np.array([1.0, 2.0, 3.0])
    assert np.isnan(ts_mean(arr, 10)).all()


def test_rolling_operators_reject_a_non_positive_window():
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ts_mean(arr, 0)
    with pytest.raises(ValueError):
        ts_std(arr, -1)
    with pytest.raises(ValueError):
        ts_rank(arr, 0)


def test_ts_zscore_is_nan_on_a_flat_window():
    arr = np.array([5.0, 5.0, 5.0, 5.0])
    out = ts_zscore(arr, 2)
    assert np.isnan(out[1:]).all()


def test_ts_rank_reports_position_within_the_window():
    arr = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    out = ts_rank(arr, 3)
    # window at t=2: [3,1,2] -> current value 2 is the middle -> rank 0.5
    assert out[2] == pytest.approx(0.5)
    # window at t=3: [1,2,5] -> current value 5 is the max -> rank 1.0
    assert out[3] == pytest.approx(1.0)
    # window at t=4: [2,5,4] -> current value 4 is the middle -> rank 0.5
    assert out[4] == pytest.approx(0.5)


def test_ts_rank_window_of_one_is_always_the_top():
    arr = np.array([3.0, -1.0, 100.0])
    out = ts_rank(arr, 1)
    assert np.array_equal(out, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------
# time-series vs cross-sectional shape enforcement
# ---------------------------------------------------------------------

def test_time_series_operators_reject_a_panel():
    panel = np.zeros((5, 3))
    for fn, args in (
        (ts_return, (10, 0)), (ts_diff, (1,)), (ts_mean, (3,)),
        (ts_std, (3,)), (ts_zscore, (3,)), (ts_rank, (3,)),
    ):
        with pytest.raises(ValueError):
            fn(panel, *args)


def test_cross_sectional_operators_reject_a_series():
    series = np.zeros(5)
    for fn in (cs_rank, cs_demean, cs_zscore):
        with pytest.raises(ValueError):
            fn(series)
    with pytest.raises(ValueError):
        winsorize(series, 0.1)


# ---------------------------------------------------------------------
# cs_rank / cs_demean / cs_zscore
# ---------------------------------------------------------------------

def test_cs_rank_scales_each_row_to_zero_one():
    panel = np.array([[1.0, 2.0, 3.0, 4.0]])
    out = cs_rank(panel)
    np.testing.assert_allclose(out[0], [0.0, 1 / 3, 2 / 3, 1.0])


def test_cs_rank_ties_share_the_average():
    panel = np.array([[1.0, 1.0, 3.0]])
    out = cs_rank(panel)
    # ranks (average method) of [1,1,3] are [1.5, 1.5, 3] -> (r-1)/(n-1)
    np.testing.assert_allclose(out[0], [0.25, 0.25, 1.0])


def test_cs_rank_of_a_lone_product_is_the_midpoint():
    panel = np.array([[np.nan, 5.0, np.nan]])
    out = cs_rank(panel)
    assert out[0, 1] == pytest.approx(0.5)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 2])


def test_cs_rank_skips_missing_products():
    panel = np.array([[1.0, np.nan, 3.0]])
    out = cs_rank(panel)
    assert np.isnan(out[0, 1])
    np.testing.assert_allclose(out[0, [0, 2]], [0.0, 1.0])


def test_cs_demean_and_zscore_ignore_missing_products():
    panel = np.array([[1.0, np.nan, 3.0]])
    demeaned = cs_demean(panel)
    assert demeaned[0, 0] == pytest.approx(-1.0)
    assert demeaned[0, 2] == pytest.approx(1.0)
    assert np.isnan(demeaned[0, 1])

    z = cs_zscore(panel)
    assert np.isnan(z[0, 1])
    assert z[0, 0] == pytest.approx(-1.0)
    assert z[0, 2] == pytest.approx(1.0)


def test_cs_zscore_is_nan_when_the_cross_section_has_no_dispersion():
    panel = np.array([[2.0, 2.0, 2.0]])
    out = cs_zscore(panel)
    assert np.isnan(out).all()


def test_all_nan_rows_do_not_warn(recwarn):
    panel = np.full((3, 4), np.nan)
    cs_rank(panel)
    cs_demean(panel)
    cs_zscore(panel)
    assert len(recwarn) == 0


# ---------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------

def test_winsorize_at_zero_is_a_passthrough():
    panel = np.array([[1.0, 2.0, 300.0]])
    out = winsorize(panel, 0.0)
    np.testing.assert_array_equal(out, panel)


def test_winsorize_clips_to_row_quantiles():
    panel = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
    out = winsorize(panel, 0.2)
    lo = np.quantile(panel[0], 0.2)
    hi = np.quantile(panel[0], 0.8)
    assert out[0].min() == pytest.approx(lo)
    assert out[0].max() == pytest.approx(hi)


def test_winsorize_rejects_a_limit_of_half():
    panel = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        winsorize(panel, 0.5)
