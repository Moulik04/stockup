"""StatsForecastQuantileModel wrapper tests. Uses a fake `_sf` (not a real StatsForecast fit —
conformal-interval sign is an implementation detail we shouldn't depend on for a fixture) to
deterministically test the post-processing clip logic itself.
"""

import pandas as pd

from reorderpoint.models.base import StatsForecastQuantileModel


class _FakeStatsForecast:
    """Returns a fixed forecast frame with deliberately negative lo-80/point/hi-80 values."""

    def forecast(self, df, h, level):
        dates = pd.date_range("2020-06-01", periods=h, freq="D")
        return pd.DataFrame(
            {
                "unique_id": ["A"] * h,
                "ds": dates,
                "Model": [-2.0] * h,
                "Model-lo-80": [-5.0] * h,
                "Model-hi-80": [1.0] * h,
            }
        )


def test_negative_conformal_bounds_are_clipped_to_zero():
    model = StatsForecastQuantileModel(model=object(), name="Model")
    model._sf = _FakeStatsForecast()
    model._train = pd.DataFrame(
        {"unique_id": ["A"], "ds": [pd.Timestamp("2020-01-01")], "y": [0.0]}
    )

    preds = model.predict_quantiles(horizon=3)

    assert (preds["p10"] >= 0).all()
    assert (preds["p50"] >= 0).all()
    assert (preds["p90"] >= 0).all()
    # confirms the clip actually engaged, not that the fixture was already non-negative
    assert (preds["p10"] == 0).all()
    assert (preds["p50"] == 0).all()
    assert (preds["p90"] == 1.0).all()  # positive input passes through unchanged
