"""The production forecaster: each series' mean demand over its last `WINDOW_SIZE` days, held flat.

This is the `MovingAverage` rung of the backtest ladder (`naive.moving_average`, statsforecast's
`WindowAverage(28)`) without the statsforecast machinery, and it forecasts the same numbers — the
equivalence is tested (`tests/test_trailing_mean.py`) and measured on the real panel
(`reports/production_model_*.md`). It is what the served system runs because, under the shipped
policy, it costs no more than LightGBM to within the noise of the comparison, needs only each
series' own demand history, fits the full 5,650-series panel in about two seconds, and
persists about 1 MB.

It has no prediction intervals. `p10` and `p90` equal `p50`, and nothing should read them: the
interval a user sees comes from `calibration.SafetyStockCalibration.calibrated_interval`, the same
residual calibration that sizes the reorder point. Not registered in `backtest.MODEL_FACTORIES` —
the ladder already contains this model, measured; this class only serves it.
"""

from __future__ import annotations

import pandas as pd

WINDOW_SIZE = 28  # 4-week moving average, as `naive.WINDOW_SIZE`


class TrailingMeanModel:
    name = "MovingAverage"

    def __init__(self, window: int = WINDOW_SIZE):
        self.window = window
        self._level: pd.Series | None = None
        self._last_date: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> None:
        ordered = train.sort_values(["series_id", "date"])
        tail = ordered.groupby("series_id", sort=True).tail(self.window)
        self._level = tail.groupby("series_id")["y"].mean().clip(lower=0)
        self._last_date = ordered.groupby("series_id")["date"].max()

    def predict_quantiles(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """One flat row per series per future day; `future_exog` is accepted and ignored."""
        if self._level is None or self._last_date is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        ids = self._level.index
        offsets = pd.to_timedelta(range(1, horizon + 1), unit="D")
        out = pd.DataFrame(
            {
                "series_id": ids.repeat(horizon),
                "date": (self._last_date.reindex(ids).to_numpy()[:, None] + offsets.to_numpy())
                .ravel()
                .astype("datetime64[ns]"),
            }
        )
        level = self._level.to_numpy().repeat(horizon)
        out["p10"] = level
        out["p50"] = level
        out["p90"] = level
        return out
