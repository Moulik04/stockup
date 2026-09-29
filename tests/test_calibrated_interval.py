"""The interval a user sees is derived from the calibration that sizes the reorder point."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from reorderpoint.calibration import INTERVAL_LEVEL, SafetyStockCalibration

LEAD_TIME = 7


def _calibration(form: str = "normal") -> SafetyStockCalibration:
    # two intermittency buckets of different spread: "wide" series are intermittent (zero rate
    # above 0.5) with large errors, "tight" ones regular with small errors
    residuals = pd.DataFrame(
        {
            "residual": [8.0, -6.0, 9.0, -7.0, 1.0, -1.0, 0.5, -0.5],
            "zero_rate": [0.9] * 4 + [0.2] * 4,
            "volume": [0.5] * 4 + [3.0] * 4,
        },
        index=pd.Index(["w1", "w2", "w3", "w4", "t1", "t2", "t3", "t4"], name="series_id"),
    )
    return SafetyStockCalibration(
        form=form,
        granularity="intermittency",
        lead_time_days=LEAD_TIME,
        model_name="Test",
        calibrated_through=pd.Timestamp("2026-01-31"),
        residuals=residuals,
    )


def _preds(series_ids, p50=2.0, days=LEAD_TIME) -> pd.DataFrame:
    dates = pd.date_range("2026-02-01", periods=days)
    return pd.DataFrame(
        [
            {"series_id": sid, "date": d, "p10": -99.0, "p50": p50, "p90": 99.0}
            for sid in series_ids
            for d in dates
        ]
    )


def test_band_and_reorder_buffer_are_read_off_the_same_sigma():
    cal = _calibration()
    sigma = cal.lead_time_std()
    band = cal.calibrated_interval(_preds(["w1", "t1"], p50=50.0))
    z_band = norm.ppf((1 + INTERVAL_LEVEL) / 2)
    for sid in ("w1", "t1"):
        day = band[band["series_id"] == sid].iloc[0]
        half = day["p90"] - day["p50"]
        # daily half-widths add in quadrature over the lead time to z * sigma ...
        assert half * np.sqrt(LEAD_TIME) == pytest.approx(z_band * sigma[sid])
    # ... and the reorder buffer is z(service level) times that same sigma
    buffer = cal.safety_stock(0.95)
    for sid in ("w1", "t1"):
        assert buffer[sid] == pytest.approx(norm.ppf(0.95) * sigma[sid])


def test_the_models_own_quantiles_never_reach_the_band():
    cal = _calibration()
    a = cal.calibrated_interval(_preds(["w1"]))
    scrambled = _preds(["w1"]).assign(p10=0.0, p90=1_000.0)
    b = cal.calibrated_interval(scrambled)
    pd.testing.assert_frame_equal(a, b)


def test_a_noisier_bucket_gets_a_wider_band():
    cal = _calibration()
    band = cal.calibrated_interval(_preds(["w1", "t1"], p50=50.0))
    width = (band["p90"] - band["p10"]).groupby(band["series_id"]).first()
    assert width["w1"] > 5 * width["t1"]


def test_lower_edge_is_clipped_at_zero():
    band = _calibration().calibrated_interval(_preds(["w1"], p50=0.1))
    assert (band["p10"] == 0.0).all()
    assert (band["p90"] > band["p50"]).all()


def test_a_series_the_calibration_never_saw_takes_the_pooled_sigma():
    cal = _calibration()
    pooled = float(np.std(cal.residuals["residual"], ddof=1))
    assert cal.lead_time_std(["new"])["new"] == pytest.approx(pooled)
    band = cal.calibrated_interval(_preds(["new"], p50=50.0)).iloc[0]
    z = norm.ppf((1 + INTERVAL_LEVEL) / 2)
    assert band["p90"] - 50.0 == pytest.approx(z * pooled / np.sqrt(LEAD_TIME))


def test_safety_stock_for_an_unseen_series_is_the_pooled_buffer_not_missing():
    cal = _calibration()
    seen_only = cal.safety_stock(0.95)
    assert "new" not in seen_only.index
    covered = cal.safety_stock(0.95, series_ids=["w1", "new"])
    assert covered["w1"] == pytest.approx(seen_only["w1"])
    pooled = norm.ppf(0.95) * float(np.std(cal.residuals["residual"], ddof=1))
    assert covered["new"] == pytest.approx(pooled)
    assert covered.notna().all()


def test_the_empirical_form_has_no_per_day_interval():
    with pytest.raises(ValueError, match="normal"):
        _calibration("empirical").calibrated_interval(_preds(["w1"]))


def test_input_frame_is_not_mutated():
    cal = _calibration()
    preds = _preds(["w1"])
    before = preds.copy()
    cal.calibrated_interval(preds)
    pd.testing.assert_frame_equal(preds, before)


def test_served_sizing_gives_an_unseen_series_a_buffer_even_when_the_model_has_no_spread():
    from reorderpoint import serve
    from reorderpoint.config import Config, CostParams

    config = Config(
        track="a",
        costs=CostParams(
            holding_cost_rate=0.02,
            stockout_penalty_per_unit=5.0,
            lead_time_days=LEAD_TIME,
            service_level_target=0.95,
        ),
    )
    cal = _calibration()
    # zero-spread quantiles (p10 == p90 == p50), as the served model emits
    preds = _preds(["w1", "new"], p50=2.0).assign(p10=2.0, p90=2.0)
    on_hand = pd.Series({"w1": 0.0, "new": 0.0})
    decisions = serve.size_decisions(preds, on_hand, config, cal)
    pooled = norm.ppf(0.95) * float(np.std(cal.residuals["residual"], ddof=1))
    assert decisions.loc["new", "safety_stock"] == pytest.approx(pooled)
    assert decisions.loc["new", "reorder_point"] == pytest.approx(2.0 * LEAD_TIME + pooled)
    assert decisions.loc["w1", "safety_stock"] == pytest.approx(cal.safety_stock(0.95)["w1"])
