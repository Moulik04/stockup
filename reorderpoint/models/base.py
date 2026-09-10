"""Common model interface: every model wraps a statsforecast model over the whole panel.

The backtester (backtest.py) treats every rung of the modelling ladder identically through
this interface: call fit(train_panel), then predict_quantiles(horizon) -> P10/P50/P90 for
every series at once. Batching all series into one statsforecast call (rather than fitting
one series at a time in a Python loop) is what makes this tractable at thousands of series.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from statsforecast import StatsForecast


class QuantileForecaster(Protocol):
    def fit(self, train: pd.DataFrame) -> None:
        """`train` has at least columns series_id, date, y (long, multi-series); models that use
        exogenous columns (price, calendar, category ids) pick what they need from `train`."""
        ...

    def predict_quantiles(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Returns columns series_id, date, p10, p50, p90 — one row per series per future date.

        `future_exog` carries the known-in-advance columns (calendar, price, category ids) for
        the forecast dates — real values during backtesting (held out, not leaked), a forward
        assumption in production. Models that only use y history (the statsforecast-based ones)
        ignore it.
        """
        ...


class StatsForecastQuantileModel:
    """Adapts one statsforecast model to the QuantileForecaster interface.

    Prediction intervals come from conformal calibration, which statsforecast applies
    uniformly across every model — including ones with no native interval support
    (e.g. WindowAverage) — so every rung of the ladder gets P10/P90 the same way.
    level=80 is the central 80% interval, i.e. P10/P90 for a symmetric predictive distribution.
    """

    def __init__(self, model, name: str, n_jobs: int = -1):
        self._model = model
        self._name = name
        self._n_jobs = n_jobs
        self._sf: StatsForecast | None = None
        self._train: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> None:
        self._train = train.rename(columns={"series_id": "unique_id", "date": "ds"})[
            ["unique_id", "ds", "y"]
        ]
        self._sf = StatsForecast(models=[self._model], freq="D", n_jobs=self._n_jobs)

    def predict_quantiles(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if self._sf is None or self._train is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        fc = self._sf.forecast(df=self._train, h=horizon, level=[80])
        out = fc.rename(columns={"unique_id": "series_id", "ds": "date"})[
            ["series_id", "date"]
        ].copy()
        # demand can't be negative — conformal intervals aren't clipped by statsforecast itself,
        # and routinely go negative for near-zero point forecasts on this intermittent panel
        # (measured: ~40% of rows for SeasonalNaive). Same clip lightgbm_global.py/deep.py apply.
        out["p10"] = np.clip(fc[f"{self._name}-lo-80"].to_numpy(), 0, None)
        out["p50"] = np.clip(fc[self._name].to_numpy(), 0, None)
        out["p90"] = np.clip(fc[f"{self._name}-hi-80"].to_numpy(), 0, None)
        return out
