"""Seasonal naive and moving average — the floor every other model must beat."""

from __future__ import annotations

from statsforecast.models import SeasonalNaive as _SeasonalNaive
from statsforecast.models import WindowAverage as _WindowAverage
from statsforecast.utils import ConformalIntervals

from reorderpoint.grain import GRAIN
from reorderpoint.models.base import StatsForecastQuantileModel

SEASON_LENGTH = GRAIN.naive_season  # 7 days (weekly cycle) daily; 52 weeks (annual) weekly
WINDOW_SIZE = GRAIN.ma_window  # 28 days daily (4 weeks); 4 weeks weekly


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
