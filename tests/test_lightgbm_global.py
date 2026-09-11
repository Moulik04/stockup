"""Tests for LightGBMGlobalModel — real (not mocked) fits on small synthetic panels, same
"exercise the real thing on tiny data" convention as test_reconcile.py.
"""

import numpy as np
import pandas as pd

from reorderpoint.models.lightgbm_global import LightGBMGlobalModel


def _panel(series_ids: list[str], n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for sid in series_ids:
        base = rng.uniform(1, 20)  # wildly different scales per series
        for d in dates:
            rows.append(
                {
                    "series_id": sid,
                    "date": d,
                    "y": max(0.0, base + rng.normal(0, base * 0.3)),
                    "price": 10.0,
                }
            )
    return pd.DataFrame(rows)


def _future_exog(panel: pd.DataFrame, series_ids: list[str], horizon: int) -> pd.DataFrame:
    last_date = panel["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    rows = []
    for sid in series_ids:
        for d in future_dates:
            rows.append({"series_id": sid, "date": d, "price": 10.0})
    return pd.DataFrame(rows)


def test_predict_quantiles_unaffected_by_unrelated_series_in_history():
    # Regression test for the performance fix: predict_quantiles used to run build_features over
    # the model's ENTIRE fit-time history on every recursive step, regardless of which series
    # were actually requested. It now scopes history to just the requested series first. Since
    # each series' features are computed independently (build_features groups by series_id, no
    # cross-series leakage), this must not change the output.
    #
    # Uses ONE trained model both times (fit once on all three series) and only varies what's in
    # self._history before predicting — comparing two separately *fitted* models here would be
    # confounded by the fact that training-data composition genuinely does change a shared
    # global model's learned trees, which isn't what this test is checking.
    horizon = 7
    panel = _panel(["A", "B", "C"], seed=1)
    model = LightGBMGlobalModel()
    model.fit(panel)
    future_exog = _future_exog(panel, ["A"], horizon)

    preds_full_history = model.predict_quantiles(horizon, future_exog=future_exog)

    model._history = model._history[model._history["series_id"] == "A"].reset_index(drop=True)
    preds_scoped_history = model.predict_quantiles(horizon, future_exog=future_exog)

    cols = ["date", "p10", "p50", "p90"]
    pd.testing.assert_frame_equal(
        preds_full_history.sort_values("date")[cols].reset_index(drop=True),
        preds_scoped_history.sort_values("date")[cols].reset_index(drop=True),
    )


def test_predict_quantiles_only_returns_requested_series():
    horizon = 5
    panel = _panel(["A", "B"], seed=2)
    model = LightGBMGlobalModel()
    model.fit(panel)
    preds = model.predict_quantiles(horizon, future_exog=_future_exog(panel, ["A"], horizon))
    assert set(preds["series_id"]) == {"A"}
    assert len(preds) == horizon
