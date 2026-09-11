"""Single global LightGBM model across all series: Tweedie point (used as P50) + quantile P10/P90.

Trained once on simple single-step-ahead features (see features.py) — zero-inflated demand
(confirmed in the EDA: 77% zero-sale days) is why the point model uses a Tweedie objective
rather than plain L2. Multi-day-ahead forecasts are produced recursively: predict day 1, feed
that prediction back into the lag/rolling features as if it were realised, predict day 2, and so
on to `horizon`. This is the standard way to get a multi-step forecast out of a model trained on
single-step features, at the cost of compounding error over the horizon — a known, accepted
trade-off, not an oversight.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from reorderpoint.features import build_features, categorical_columns, feature_columns

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)

# Only the trailing window needed to recompute lag_28 / roll_28 at each recursive step —
# keeps every recursive build_features() call cheap regardless of total training history length.
MAX_HISTORY_DAYS = 90


class LightGBMGlobalModel:
    def __init__(self, lo_alpha: float = 0.13, hi_alpha: float = 0.87):
        """lo_alpha/hi_alpha default to 0.13/0.87, not the "obvious" 0.10/0.90: a backtest run
        with 0.10/0.90 measured 85.6% empirical coverage against an 80% target. This project's
        quantile regressors run a bit wide, so the nominal target is narrowed inward until
        empirical coverage lands near 80%.
        """
        self._lo_alpha = lo_alpha
        self._hi_alpha = hi_alpha
        self._models: dict[str, lgb.LGBMRegressor] = {}
        self._history: pd.DataFrame | None = None
        self._feature_cols: list[str] | None = None
        self._categorical_cols: list[str] | None = None

    def fit(self, train: pd.DataFrame) -> None:
        featured = build_features(train)
        self._feature_cols = feature_columns(featured)
        self._categorical_cols = categorical_columns(featured)

        fit_rows = featured.dropna(subset=["lag_1"]).copy()
        for col in self._categorical_cols:
            fit_rows[col] = fit_rows[col].astype("category")
        X = fit_rows[self._feature_cols]
        y = fit_rows["y"]

        point = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, **LGB_PARAMS)
        point.fit(X, y, categorical_feature=self._categorical_cols)

        q10 = lgb.LGBMRegressor(objective="quantile", alpha=self._lo_alpha, **LGB_PARAMS)
        q10.fit(X, y, categorical_feature=self._categorical_cols)

        q90 = lgb.LGBMRegressor(objective="quantile", alpha=self._hi_alpha, **LGB_PARAMS)
        q90.fit(X, y, categorical_feature=self._categorical_cols)

        self._models = {"p50": point, "p10": q10, "p90": q90}

        keep_cols = [c for c in train.columns if c != "y"] + ["y"]
        self._history = (
            train[keep_cols]
            .sort_values(["series_id", "date"])
            .groupby("series_id", group_keys=False, sort=False)
            .tail(MAX_HISTORY_DAYS)
            .reset_index(drop=True)
        )

    def predict_quantiles(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if self._models is None or self._history is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        if future_exog is None or future_exog.empty:
            raise ValueError(
                "LightGBMGlobalModel.predict_quantiles requires future_exog: calendar/price/"
                "category columns for the forecast dates (known in advance, not the target y)"
            )

        # Scope history down to only the requested series before the recursive loop. Each
        # series' lag/rolling features are computed independently in build_features (grouped by
        # series_id, no cross-series leakage), so this changes nothing about the output — but it
        # matters a lot for cost: the production model's self._history covers the entire training
        # panel (all series fit on), and a live serving request typically asks for one or two.
        # Recomputing features across the whole panel on every recursive step just to get one new
        # row for the requested series is the difference between a dashboard click taking
        # seconds vs. the better part of a minute.
        requested_series = future_exog["series_id"].unique()
        history = self._history[self._history["series_id"].isin(requested_series)].copy()
        forecast_dates = sorted(future_exog["date"].unique())
        rows_out = []

        for forecast_date in forecast_dates:
            exog_today = future_exog[future_exog["date"] == forecast_date].copy()
            placeholder = exog_today.copy()
            placeholder["y"] = np.nan

            working = pd.concat([history, placeholder], ignore_index=True, sort=False)
            featured = build_features(working)
            today = featured[featured["date"] == forecast_date].copy()
            for col in self._categorical_cols:
                today[col] = today[col].astype("category")
            X = today[self._feature_cols]

            p10_raw = self._models["p10"].predict(X)
            p50_raw = self._models["p50"].predict(X)
            p90_raw = self._models["p90"].predict(X)

            # demand can't be negative; enforce p10 <= p50 <= p90 in case the three independently
            # trained models cross (a known quantile-regression artifact, not a bug).
            p50 = np.clip(p50_raw, 0, None)
            p10 = np.clip(np.minimum(p10_raw, p50_raw), 0, None)
            p90 = np.clip(np.maximum(p90_raw, p50_raw), 0, None)

            series_ids_today = today["series_id"].to_numpy()
            rows_out.append(
                pd.DataFrame(
                    {
                        "series_id": series_ids_today,
                        "date": forecast_date,
                        "p10": p10,
                        "p50": p50,
                        "p90": p90,
                    }
                )
            )

            p50_by_series = pd.Series(p50, index=series_ids_today)
            realized = exog_today.copy()
            realized["y"] = realized["series_id"].map(p50_by_series)
            history = pd.concat([history, realized], ignore_index=True, sort=False)
            history = (
                history.sort_values(["series_id", "date"])
                .groupby("series_id", group_keys=False, sort=False)
                .tail(MAX_HISTORY_DAYS)
                .reset_index(drop=True)
            )

        return pd.concat(rows_out, ignore_index=True)


def lightgbm_global(horizon: int) -> LightGBMGlobalModel:
    return LightGBMGlobalModel()
