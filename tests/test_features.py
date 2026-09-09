"""Leakage tests: perturbing y after a cutoff date must not change features computed at or
before that cutoff. Mandatory before any feature ships.
"""

import pandas as pd
import pytest

from reorderpoint.features import build_features, categorical_columns, feature_columns


def _series_panel(values: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {"series_id": ["A"] * len(values), "date": dates, "y": values, "price": 10.0}
    )


def test_features_at_cutoff_unaffected_by_perturbing_future_y():
    base_values = [float(v) for v in range(1, 41)]  # 40 days: enough for lag_28 / roll_28
    cutoff_idx = 20  # 0-indexed day 21

    perturbed_values = base_values.copy()
    for i in range(cutoff_idx, len(perturbed_values)):
        perturbed_values[i] = 999.0

    features_a = build_features(_series_panel(base_values))
    features_b = build_features(_series_panel(perturbed_values))

    cols = feature_columns(features_a)
    row_a = features_a.loc[cutoff_idx, cols]
    row_b = features_b.loc[cutoff_idx, cols]
    pd.testing.assert_series_equal(row_a, row_b, check_names=False)


def test_features_before_cutoff_also_unaffected():
    base_values = [float(v) for v in range(1, 41)]
    perturbed_values = base_values.copy()
    for i in range(20, len(perturbed_values)):
        perturbed_values[i] = 999.0

    features_a = build_features(_series_panel(base_values))
    features_b = build_features(_series_panel(perturbed_values))

    cols = feature_columns(features_a)
    for idx in (0, 5, 10, 19):
        pd.testing.assert_series_equal(
            features_a.loc[idx, cols], features_b.loc[idx, cols], check_names=False
        )


def test_lag_1_equals_previous_day_y():
    panel = _series_panel([10.0, 20.0, 30.0, 40.0])
    features = build_features(panel)
    assert features.loc[1, "lag_1"] == 10.0
    assert features.loc[2, "lag_1"] == 20.0
    assert pd.isna(features.loc[0, "lag_1"])


def test_rolling_mean_excludes_current_day():
    panel = _series_panel([10.0] * 10 + [1000.0])  # last day is a huge same-day outlier
    features = build_features(panel)
    assert features.iloc[-1]["roll_mean_7"] == pytest.approx(10.0)


def test_rolling_zero_rate_hand_computed():
    panel = _series_panel([0.0, 0.0, 5.0, 0.0, 5.0])
    features = build_features(panel)
    # window at index 4 covers shifted y = [NaN, 0, 0, 5, 0] (the leading NaN has no prior day).
    # (s == 0) treats NaN as False ("not zero"), so the window is [F, T, T, F, T] -> 3/5 = 0.6.
    assert features.loc[4, "roll_zero_rate_7"] == pytest.approx(0.6)


def test_price_change_uses_only_known_prices():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * 10,
            "date": dates,
            "y": [1.0] * 10,
            "price": [10.0] * 7 + [12.0] * 3,
        }
    )
    features = build_features(panel)
    # day index 7: price=12, price 7 days earlier (index 0) = 10 -> +20%
    assert features.loc[7, "price_change_7"] == pytest.approx(0.2)


def test_feature_columns_omits_price_change_when_no_price():
    panel = pd.DataFrame(
        {"series_id": ["A"] * 5, "date": pd.date_range("2020-01-01", periods=5), "y": [1.0] * 5}
    )
    features = build_features(panel)
    cols = feature_columns(features)
    assert "price_change_7" not in cols
    assert "lag_1" in cols


def test_categorical_columns_only_returns_present_ones():
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * 3,
            "date": pd.date_range("2020-01-01", periods=3),
            "y": [1.0, 2.0, 3.0],
            "item_id": ["I1"] * 3,
        }
    )
    assert categorical_columns(panel) == ["item_id"]


def test_multi_series_lags_dont_leak_across_series():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": ["A", "A", "A", "B", "B", "B"],
            "date": list(dates) * 2,
            "y": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0],
        }
    )
    features = build_features(panel)
    b_first_row = features[(features["series_id"] == "B") & (features["date"] == dates[0])]
    assert pd.isna(b_first_row["lag_1"].iloc[0])  # not contaminated by series A's last value
