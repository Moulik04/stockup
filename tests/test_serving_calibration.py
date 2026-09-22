"""The served path must size safety stock the way the backtest harness does.

Serving used `z * σ` with σ from the model's own P10/P90 width — the method Phase 4 showed loses —
while every result in the README was produced by the calibrated harness. These tests pin the two
to one rule: the same series and forecast give the same reorder point through
`ablate_safety_stock.run_cell` (the harness) and `serve.size_decisions` / `POST /reorder`
(serving), and serving refuses to fall back to the raw method when no calibration exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from scipy.stats import norm

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import backtest as bt
from reorderpoint import safety_stock as ss
from reorderpoint import serve
from reorderpoint.calibration import SafetyStockCalibration
from reorderpoint.config import Config, CostParams

LEAD = 7
TARGET = 0.95
N_SERIES = 12


def _costs(**over) -> CostParams:
    base = dict(
        holding_cost_rate=0.001,
        stockout_penalty_per_unit=5.0,
        lead_time_days=LEAD,
        service_level_target=TARGET,
    )
    base.update(over)
    return CostParams(**base)


def _config(**over) -> Config:
    return Config(track="a", track_b_data_path=None, costs=_costs(**over))


def _panel() -> pd.DataFrame:
    """Half the series regular (few zero days), half intermittent (mostly zero) — both
    intermittency buckets populated, long enough for the harness's four 28-day folds."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=240, freq="D")
    rows = []
    for i in range(N_SERIES):
        p_zero = 0.1 if i % 2 == 0 else 0.8
        y = np.where(rng.random(len(dates)) < p_zero, 0.0, rng.integers(1, 6, len(dates)))
        rows += [
            {"series_id": f"S{i:02d}", "date": d, "y": float(v), "price": 5.0 + i}
            for d, v in zip(dates, y, strict=True)
        ]
    return pd.DataFrame(rows)


def _forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in bt.make_folds(panel):
        for d in pd.date_range(fold.test_start, fold.test_end):
            for i in range(N_SERIES):
                p50 = 0.6 + 0.35 * i
                rows.append(
                    {
                        "fold": fold.index,
                        "model": "LightGBM",
                        "series_id": f"S{i:02d}",
                        "date": d,
                        "p10": p50 * 0.5,
                        "p50": p50,
                        "p90": p50 * 1.7,
                    }
                )
    return pd.DataFrame(rows)


def _calib(panel: pd.DataFrame) -> pd.DataFrame:
    """The harness's fold-1 calibration table: residual + the two bucketing variables."""
    rng = np.random.default_rng(1)
    zero_rates, volumes = ss.train_stats(
        panel[panel["date"] < panel["date"].max() - pd.Timedelta(days=112)]
    )
    ids = sorted(zero_rates.index)
    return pd.DataFrame(
        {
            "model": "LightGBM",
            "series_id": ids,
            "residual": rng.normal(0, 1, len(ids)) * np.where(zero_rates[ids] > 0.5, 2.0, 6.0),
            "zero_rate": zero_rates[ids].to_numpy(),
            "volume": volumes[ids].to_numpy(),
        }
    )


def _calibration(calib: pd.DataFrame) -> SafetyStockCalibration:
    return SafetyStockCalibration(
        form=ss.DEFAULT_SCHEME[0],
        granularity=ss.DEFAULT_SCHEME[1],
        lead_time_days=LEAD,
        model_name="LightGBM",
        calibrated_through=pd.Timestamp("2020-08-27"),
        residuals=calib.set_index("series_id")[["residual", "zero_rate", "volume"]],
    )


def test_serving_and_harness_produce_the_same_reorder_point(monkeypatch):
    monkeypatch.setattr(bt, "MODEL_FACTORIES", {"LightGBM": None})
    panel, calib_df = _panel(), None
    forecasts = _forecasts(panel)
    calib_df = _calib(panel)

    _, detail, _ = ab.run_cell(
        panel, forecasts, calib_df, *ss.DEFAULT_SCHEME, _costs(), n_boot=2, service_level=TARGET
    )
    harness = detail[detail["fold"] == 1].set_index("series_id")["reorder_point"]
    assert len(harness) == N_SERIES

    preds = forecasts[forecasts["fold"] == 1][["series_id", "date", "p10", "p50", "p90"]]
    served = serve.size_decisions(
        preds,
        pd.Series(0.0, index=sorted(preds["series_id"].unique())),
        _config(),
        _calibration(calib_df),
    )["reorder_point"]

    pd.testing.assert_series_equal(
        served.sort_index(), harness.sort_index(), check_names=False, rtol=1e-12
    )


def test_calibrated_sizing_differs_from_the_raw_quantile_method():
    """The regression this file exists for: the two methods disagree, so wiring matters."""
    panel = _panel()
    forecasts = _forecasts(panel)
    preds = forecasts[forecasts["fold"] == 1][["series_id", "date", "p10", "p50", "p90"]]
    on_hand = pd.Series(0.0, index=sorted(preds["series_id"].unique()))
    calibrated = serve.size_decisions(preds, on_hand, _config(), _calibration(_calib(panel)))
    from reorderpoint import decision as dec

    raw = dec.compute_decisions(preds, on_hand, LEAD, TARGET)
    assert not np.allclose(calibrated["safety_stock"], raw["safety_stock"])


def _stub_calibration() -> SafetyStockCalibration:
    # two regular series (zero rate 0.2) with residuals +/-6 -> bucket std 8.485, z*std = 13.96
    resid = pd.DataFrame(
        {"residual": [6.0, -6.0], "zero_rate": [0.2, 0.2], "volume": [2.0, 2.0]},
        index=pd.Index(["A", "B"], name="series_id"),
    )
    return SafetyStockCalibration(
        form="normal",
        granularity="intermittency",
        lead_time_days=LEAD,
        model_name="LightGBM",
        calibrated_through=pd.Timestamp("2020-02-29"),
        residuals=resid,
    )


class _ConstantModel:
    def predict_quantiles(self, horizon, future_exog=None):
        dates = sorted(future_exog["date"].unique())
        return pd.DataFrame(
            [
                {"series_id": sid, "date": d, "p10": 1.0, "p50": 2.0, "p90": 3.0}
                for sid in future_exog["series_id"].unique()
                for d in dates
            ]
        )


def _small_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    return pd.DataFrame(
        [
            {"series_id": s, "date": d, "y": float(i % 5), "price": 10.0}
            for s in ("A", "B")
            for i, d in enumerate(dates)
        ]
    )


@pytest.fixture
def app_client():
    app = serve.app
    app.dependency_overrides[serve.get_model] = lambda: _ConstantModel()
    app.dependency_overrides[serve.get_history] = _small_panel
    app.dependency_overrides[serve.get_config] = lambda: _config()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_reorder_endpoint_sizes_from_the_calibration_not_the_interval(app_client):
    serve.app.dependency_overrides[serve.get_calibration] = _stub_calibration
    row = app_client.post("/reorder", json={"series_ids": ["A"]}).json()[0]
    z = norm.ppf(TARGET)
    assert row["safety_stock"] == pytest.approx(z * np.std([6.0, -6.0], ddof=1))
    # the raw method would have given z * sqrt(7) * (3-1)/(2*Z80) ~ 3.4 for this forecast
    assert row["safety_stock"] > 10
    assert row["reorder_point"] == pytest.approx(14.0 + row["safety_stock"])


def test_reorder_without_a_calibration_is_503_never_a_silent_fallback(app_client, monkeypatch):
    def missing():
        raise RuntimeError("no safety-stock calibration")

    monkeypatch.setattr(serve, "_load_calibration", missing)
    resp = app_client.post("/reorder", json={"series_ids": ["A"]})
    assert resp.status_code == 503
    assert "calibration" in resp.json()["detail"]


def test_reorder_rejects_a_calibration_for_a_different_lead_time(app_client):
    serve.app.dependency_overrides[serve.get_calibration] = _stub_calibration
    serve.app.dependency_overrides[serve.get_config] = lambda: _config(lead_time_days=14)
    resp = app_client.post("/reorder", json={"series_ids": ["A"]})
    assert resp.status_code == 409


def test_get_calibration_maps_a_missing_file_to_503(monkeypatch):
    def missing():
        raise RuntimeError("nope")

    monkeypatch.setattr(serve, "_load_calibration", missing)
    with pytest.raises(HTTPException) as exc:
        serve.get_calibration()
    assert exc.value.status_code == 503


def test_fit_calibration_holds_out_the_last_lead_window_and_matches_the_harness_sizing():
    """`fit_calibration` reproduces the harness's calibration procedure: residuals are realised
    minus predicted lead-time demand over a held-out final window, and the resulting per-series
    safety stock is exactly `per_series_safety_stock` of that table."""
    from reorderpoint import calibration as cal

    panel = _panel()

    class ConstantModel:
        def fit(self, train):
            self.train_end = train["date"].max()

        def predict_quantiles(self, horizon, future_exog=None):
            rows = [
                {"series_id": sid, "date": d, "p10": 0.5, "p50": 1.0, "p90": 2.0}
                for sid in future_exog["series_id"].unique()
                for d in sorted(future_exog["date"].unique())
            ]
            return pd.DataFrame(rows)

    fitted = cal.fit_calibration(panel, ConstantModel, LEAD)
    last = panel["date"].max()
    assert fitted.calibrated_through == last and fitted.lead_time_days == LEAD
    window = panel[panel["date"] >= last - pd.Timedelta(days=LEAD - 1)]
    expected = window.groupby("series_id")["y"].sum() - LEAD * 1.0
    got = fitted.residuals["residual"].sort_index()
    pd.testing.assert_series_equal(got, expected.sort_index(), check_names=False)
    direct, _ = ss.per_series_safety_stock(
        fitted.residuals["residual"],
        fitted.residuals["zero_rate"],
        fitted.residuals["volume"],
        TARGET,
        form="normal",
        granularity="intermittency",
        n_boot=2,
    )
    pd.testing.assert_series_equal(fitted.safety_stock(TARGET), direct)
    # the artifact round-trips
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.joblib"
        cal.save(fitted, path)
        pd.testing.assert_frame_equal(cal.load(path).residuals, fitted.residuals)


def test_serving_and_harness_agree_on_the_order_up_to_level_too(monkeypatch):
    """The lot multiple is a policy parameter shared by the harness and serving: with the same
    multiple both must produce the same reorder point *and* the same order-up-to level."""
    monkeypatch.setattr(bt, "MODEL_FACTORIES", {"LightGBM": None})
    panel = _panel()
    forecasts, calib_df = _forecasts(panel), _calib(panel)
    lot = 2.0

    _, detail, _ = ab.run_cell(
        panel,
        forecasts,
        calib_df,
        *ss.DEFAULT_SCHEME,
        _costs(),
        n_boot=2,
        service_level=TARGET,
        lot_multiple=lot,
    )
    harness = detail[detail["fold"] == 1].set_index("series_id")

    preds = forecasts[forecasts["fold"] == 1][["series_id", "date", "p10", "p50", "p90"]]
    served = serve.size_decisions(
        preds,
        pd.Series(0.0, index=sorted(preds["series_id"].unique())),
        Config(track="a", track_b_data_path=None, costs=_costs(lot_multiple=lot)),
        _calibration(calib_df),
    )
    for col in ("reorder_point", "order_up_to"):
        pd.testing.assert_series_equal(
            served[col].sort_index(), harness[col].sort_index(), check_names=False, rtol=1e-12
        )
    assert (served["order_up_to"] > served["reorder_point"]).all()
