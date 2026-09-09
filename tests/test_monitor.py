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
