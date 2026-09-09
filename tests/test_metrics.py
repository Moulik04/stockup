"""Hand-computed checks for each metric formula."""

import numpy as np
import pytest

from reorderpoint.metrics import coverage, mase, pinball_loss, rmsse, weighted_average


def test_mase_hand_computed():
    y_train = np.arange(1, 11, dtype=float)  # 1..10, lag-1 diffs are all 1 -> scale=1
    y_true = np.array([10.0, 11.0, 12.0])
    y_pred = np.array([9.0, 11.0, 15.0])
    # |errors| = [1, 0, 3], mean = 4/3
    assert mase(y_true, y_pred, y_train, season_length=1) == pytest.approx(4 / 3)


def test_rmsse_hand_computed():
    y_train = np.arange(1, 11, dtype=float)
    y_true = np.array([10.0, 11.0, 12.0])
    y_pred = np.array([9.0, 11.0, 15.0])
    # squared errors = [1, 0, 9], mean = 10/3, scale=1 -> sqrt(10/3)
    assert rmsse(y_true, y_pred, y_train, season_length=1) == pytest.approx(np.sqrt(10 / 3))


def test_mase_zero_scale_returns_nan():
    y_train = np.ones(10)  # constant -> in-sample naive error is 0
    y_true = np.array([1.0])
    y_pred = np.array([2.0])
    assert np.isnan(mase(y_true, y_pred, y_train, season_length=1))


def test_pinball_loss_median_equals_half_mae():
    y_true = np.array([10.0])
    y_pred = np.array([9.0])
    assert pinball_loss(y_true, y_pred, 0.5) == pytest.approx(0.5)


def test_pinball_loss_asymmetric_quantiles():
    y_true = np.array([10.0])
    y_pred_low = np.array([11.0])  # over-forecast, diff = -1
    assert pinball_loss(y_true, y_pred_low, 0.1) == pytest.approx(0.9)


def test_coverage():
    y_true = np.array([5.0, 15.0, 10.0])
    lo = np.zeros(3)
    hi = np.full(3, 10.0)
    assert coverage(y_true, lo, hi) == pytest.approx(2 / 3)


def test_weighted_average_ignores_nan():
    values = np.array([1.0, np.nan, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert weighted_average(values, weights) == pytest.approx(2.0)
