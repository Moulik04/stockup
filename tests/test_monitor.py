"""Hand-computed checks for realised-error tracking and drift detection (Phase 6)."""

import numpy as np
import pandas as pd
import pytest

from reorderpoint import monitor as mon


def test_realised_error_hand_computed():
    log = pd.DataFrame(
        {
            "series_id": ["A", "A", "B", "B"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2),
            "y": [10.0, 12.0, 5.0, 5.0],
            "forecast": [8.0, 12.0, 5.0, 9.0],
        }
    )
    out = mon.realised_error(log)
    assert out.loc[(out["series_id"] == "A") & (out["date"] == "2020-01-01"), "abs_error"].iloc[
        0
    ] == pytest.approx(2.0)
    assert out.loc[(out["series_id"] == "B") & (out["date"] == "2020-01-02"), "abs_error"].iloc[
        0
    ] == pytest.approx(4.0)


def test_rolling_mase_hand_computed():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    log = pd.DataFrame(
        {
            "series_id": ["A"] * 6,
            "date": dates,
            "y": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "forecast": [10.0, 10.0, 10.0, 15.0, 15.0, 15.0],  # errors: 0,0,0,5,5,5
        }
    )
    errors = mon.realised_error(log)
    out = mon.rolling_mase(errors, scale=2.0, window=3)
    # last 3-day window (days 4,5,6): mean abs error = 5 -> mase = 5/2 = 2.5
    last_row = out[out["date"] == dates[-1]].iloc[0]
    assert last_row["rolling_mase"] == pytest.approx(2.5)
    # first window (only day 1 available): mean abs error = 0 -> mase = 0
    first_row = out[out["date"] == dates[0]].iloc[0]
    assert first_row["rolling_mase"] == pytest.approx(0.0)


def test_rolling_mase_uses_each_series_own_scale():
    # Two series with the same rolling abs-error (5.0) but different scales — if `scale` were
    # applied as one shared value (the bug), both would get the same rolling_mase, defeating
    # MASE's purpose as a scale-free metric comparable across series of different volumes.
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    log = pd.DataFrame(
        {
            "series_id": ["LOW_VOL", "LOW_VOL", "HIGH_VOL", "HIGH_VOL"],
            "date": list(dates) * 2,
            "y": [10.0, 10.0, 100.0, 100.0],
            "forecast": [10.0, 15.0, 100.0, 105.0],  # abs error 0, 5 for both series
        }
    )
    errors = mon.realised_error(log)
    scale = pd.Series({"LOW_VOL": 1.0, "HIGH_VOL": 10.0})
    out = mon.rolling_mase(errors, scale=scale, window=2)

    low_vol_last = out[(out["series_id"] == "LOW_VOL") & (out["date"] == dates[-1])].iloc[0]
    high_vol_last = out[(out["series_id"] == "HIGH_VOL") & (out["date"] == dates[-1])].iloc[0]
    # both have rolling mean abs error = 2.5, but divided by their own scale
    assert low_vol_last["rolling_mase"] == pytest.approx(2.5)  # 2.5 / 1.0
    assert high_vol_last["rolling_mase"] == pytest.approx(0.25)  # 2.5 / 10.0


def test_flag_error_spikes_detects_jump_above_threshold():
    rolling = pd.DataFrame(
        {
            "series_id": ["A", "B"],
            "date": pd.to_datetime(["2020-01-10", "2020-01-10"]),
            "rolling_mase": [3.5, 0.8],
        }
    )
    flagged = mon.flag_error_spikes(rolling, threshold=1.0)
    assert flagged == ["A"]


def test_detect_input_drift_flags_shifted_series():
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame(
        {
            "series_id": ["STABLE"] * 200 + ["DRIFTED"] * 200,
            "y": np.concatenate([rng.normal(10, 1, 200), rng.normal(10, 1, 200)]),
        }
    )
    recent = pd.DataFrame(
        {
            "series_id": ["STABLE"] * 20 + ["DRIFTED"] * 20,
            # STABLE stays near its baseline mean; DRIFTED jumps by +8 std devs
            "y": np.concatenate([rng.normal(10, 1, 20), rng.normal(18, 1, 20)]),
        }
    )
    flagged = mon.detect_input_drift(baseline, recent, z_threshold=3.0)
    assert "DRIFTED" in flagged
    assert "STABLE" not in flagged


def test_detect_input_drift_flags_zero_variance_baseline_that_shifts():
    # A series that was exactly 0 throughout its baseline window (std=0, common on this
    # ~77%-zero-sale-day panel) then starts selling — a real, common drift case this project
    # needs to catch, not a synthetic edge case.
    baseline = pd.DataFrame({"series_id": ["A"] * 50 + ["STILL_ZERO"] * 50, "y": [0.0] * 100})
    recent = pd.DataFrame(
        {"series_id": ["A"] * 10 + ["STILL_ZERO"] * 10, "y": [5.0] * 10 + [0.0] * 10}
    )
    flagged = mon.detect_input_drift(baseline, recent, z_threshold=3.0)
    assert "A" in flagged
    assert "STILL_ZERO" not in flagged
