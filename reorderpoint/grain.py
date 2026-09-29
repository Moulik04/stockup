"""The period a track is measured in: one object for everything that used to assume "a day".

Track A (M5) is daily; Track B (UCI Online Retail II) is weekly. The simulation, the calibration and
the bootstrap only ever count periods, so they did not care; what did care was a handful of places
that quietly meant days: fold arithmetic, a model's season length and pandas frequency, its
averaging window, the horizon, LightGBM's lags, and the defaults for lead time and holding rate.
They now read a `Grain`.

Track A's `DAILY` holds exactly the constants those places used to hard-code, so nothing about
Track A moves (`tests/test_track_a_regression.py` checks the numbers). The grain is chosen once per
process from `REORDERPOINT_TRACK` at import, like the panel path and the caches beside it, so a
process cannot mix the two.

Weekly choices, registered before any Track B model was run (`docs/track_b.md`):

- horizon 13 weeks, 4 folds, so the folds cover the final 52 weeks;
- SeasonalNaive uses the 52-week season; AutoETS/AutoTheta are non-seasonal, because no training
  window ever holds the two full annual cycles a seasonal model of period 52 needs (the longest is
  91 weeks). That is the registered fallback, and it is stated wherever those models' results are;
- the moving average is 4 weeks (the daily 28 days);
- MASE is scaled by the one-step naive error: a 52-week seasonal scale would be undefined for the
  52-week first fold;
- the default lead time is 2 weeks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Grain:
    name: str  # "day" | "week"
    period: pd.Timedelta
    freq: str  # pandas / statsforecast frequency alias
    periods_per_year: int
    horizon: int  # forecast horizon and fold length, in periods
    n_folds: int
    mase_lag: int  # lag of the naive error MASE is scaled by
    naive_season: int  # SeasonalNaive's season, in periods
    model_season: int  # AutoETS / AutoTheta season; 1 means non-seasonal
    ma_window: int  # the moving-average rung's window, in periods
    default_lead_time: int  # in periods
    n_series_sample: int | None  # None: every series
    lags: tuple[int, ...]  # LightGBM lag features
    rolling_windows: tuple[int, ...]
    price_change_lag: int
    lgb_history: int  # trailing rows LightGBM's recursive rollout keeps per series
    unit: str  # "days" | "weeks", for text

    def periods_in(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Periods from `start` to `end` inclusive (both are period starts)."""
        return int((end - start) / self.period) + 1

    def before(self, ts: pd.Timestamp, n: int) -> pd.Timestamp:
        return ts - n * self.period

    def annual_to_period(self, annual_rate: float) -> float:
        """The simulation accrues holding once per period; simple, not compounded."""
        return annual_rate / self.periods_per_year

    def period_to_annual(self, period_rate: float) -> float:
        return period_rate * self.periods_per_year


DAILY = Grain(
    name="day",
    period=pd.Timedelta(days=1),
    freq="D",
    periods_per_year=365,
    horizon=28,
    n_folds=4,
    mase_lag=7,
    naive_season=7,
    model_season=7,
    ma_window=28,
    default_lead_time=7,
    n_series_sample=400,
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 28),
    price_change_lag=7,
    lgb_history=90,
    unit="days",
)

WEEKLY = Grain(
    name="week",
    period=pd.Timedelta(weeks=1),
    freq="W-MON",
    periods_per_year=52,
    horizon=13,
    n_folds=4,
    mase_lag=1,
    naive_season=52,
    model_season=1,
    ma_window=4,
    default_lead_time=2,
    n_series_sample=None,
    lags=(1, 2, 4, 13),
    rolling_windows=(4, 13),
    price_change_lag=4,
    lgb_history=30,
    unit="weeks",
)


def for_track(track: str) -> Grain:
    if track == "a":
        return DAILY
    if track == "b":
        return WEEKLY
    raise ValueError(f"REORDERPOINT_TRACK must be 'a' or 'b', got {track!r}")


GRAIN = for_track(os.environ.get("REORDERPOINT_TRACK", "a").lower())
