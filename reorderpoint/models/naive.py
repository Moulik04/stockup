"""Seasonal naive and moving average — the floor every other model must beat."""

from __future__ import annotations

from statsforecast.models import SeasonalNaive as _SeasonalNaive
from statsforecast.models import WindowAverage as _WindowAverage
from statsforecast.utils import ConformalIntervals

from reorderpoint.models.base import StatsForecastQuantileModel

SEASON_LENGTH = 7  # weekly seasonality in daily M5 data
WINDOW_SIZE = 28  # 4-week moving average


def seasonal_naive(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _SeasonalNaive(season_length=SEASON_LENGTH, prediction_intervals=ci),
        name="SeasonalNaive",
    )


def moving_average(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _WindowAverage(window_size=WINDOW_SIZE, prediction_intervals=ci),
        name="WindowAverage",
    )
