"""ETS and Theta via statsforecast — fast, free, strong per-series baselines."""

from __future__ import annotations

from statsforecast.models import AutoETS as _AutoETS
from statsforecast.models import AutoTheta as _AutoTheta
from statsforecast.utils import ConformalIntervals

from reorderpoint.models.base import StatsForecastQuantileModel

SEASON_LENGTH = 7  # weekly seasonality in daily M5 data


def auto_ets(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _AutoETS(season_length=SEASON_LENGTH, prediction_intervals=ci),
        name="AutoETS",
    )


def auto_theta(horizon: int) -> StatsForecastQuantileModel:
    ci = ConformalIntervals(n_windows=2, h=horizon)
    return StatsForecastQuantileModel(
        _AutoTheta(season_length=SEASON_LENGTH, prediction_intervals=ci),
        name="AutoTheta",
    )
