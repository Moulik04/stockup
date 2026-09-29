"""StatsForecastQuantileModel wrapper tests. Uses a fake `_sf` (not a real StatsForecast fit —
conformal-interval sign is an implementation detail we shouldn't depend on for a fixture) to
deterministically test the post-processing clip logic itself.
"""

import numpy as np
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


def test_a_seasonal_model_falls_back_to_its_non_seasonal_form_on_too_short_a_window():
    from reorderpoint.models import naive

    def panel(n):
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        y = np.tile(np.arange(52, dtype=float), 3)[:n]  # a sawtooth with a period of 52
        return pd.DataFrame({"series_id": "a", "date": dates, "y": y})

    # a 52-period season (Track B's annual weekly one) built on the daily grid the tests run on
    model = naive.StatsForecastQuantileModel(
        naive._SeasonalNaive(season_length=52),
        name="SeasonalNaive",
        n_jobs=1,
        min_history=52,
        fallback=naive._SeasonalNaive(season_length=1),
    )
    short = panel(50)
    model.fit(short)
    assert model.used_fallback
    flat = model.predict_quantiles(4)["p50"].to_numpy()
    assert np.allclose(flat, short["y"].iloc[-1])  # the last value, repeated: plain naive

    long = panel(60)
    model.fit(long)
    assert not model.used_fallback
    seasonal = model.predict_quantiles(4)["p50"].to_numpy()
    assert np.allclose(seasonal, long["y"].iloc[8:12].to_numpy())  # one season back, step for step
