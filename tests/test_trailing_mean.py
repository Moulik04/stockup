"""The served model is the ladder's MovingAverage rung, held flat and with no dependencies."""

import numpy as np
import pandas as pd
import pytest

from reorderpoint.models import naive
from reorderpoint.models.trailing_mean import WINDOW_SIZE, TrailingMeanModel


def _panel(n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {"series_id": sid, "date": d, "y": float(rng.poisson(lam))}
            for sid, lam in (("a", 0.3), ("b", 4.0))
            for d in dates
        ]
    )


def test_window_matches_the_ladders_moving_average():
    assert WINDOW_SIZE == naive.WINDOW_SIZE


def test_forecast_is_the_mean_of_the_last_window_held_flat():
    panel = _panel()
    model = TrailingMeanModel()
    model.fit(panel)
    out = model.predict_quantiles(10)
    assert len(out) == 20
    for sid, group in panel.groupby("series_id"):
        expected = group.sort_values("date")["y"].tail(WINDOW_SIZE).mean()
        fc = out[out["series_id"] == sid]
        assert fc["p50"].tolist() == pytest.approx([expected] * 10)


def test_dates_continue_from_each_series_last_date():
    panel = _panel()
    model = TrailingMeanModel()
    model.fit(panel)
    out = model.predict_quantiles(3)
    last = panel["date"].max()
    assert sorted(out["date"].unique()) == list(
        pd.date_range(last + pd.Timedelta(days=1), periods=3)
    )


def test_matches_statsforecast_window_average():
    panel = _panel(90)
    horizon = 14
    ours = TrailingMeanModel()
    ours.fit(panel)
    ref = naive.moving_average(horizon)
    ref._n_jobs = 1
    ref.fit(panel)
    a = ours.predict_quantiles(horizon).sort_values(["series_id", "date"])
    b = ref.predict_quantiles(horizon).sort_values(["series_id", "date"])
    assert a["series_id"].tolist() == b["series_id"].tolist()
    assert a["p50"].to_numpy() == pytest.approx(b["p50"].to_numpy(), abs=1e-5)


def test_has_no_intervals_of_its_own_and_ignores_future_exog():
    panel = _panel()
    model = TrailingMeanModel()
    model.fit(panel)
    plain = model.predict_quantiles(5)
    with_exog = model.predict_quantiles(5, future_exog=pd.DataFrame({"price": [1.0]}))
    assert (plain["p10"] == plain["p50"]).all() and (plain["p90"] == plain["p50"]).all()
    pd.testing.assert_frame_equal(plain, with_exog)


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        TrailingMeanModel().predict_quantiles(3)
