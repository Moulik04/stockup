"""Point-in-time-safe feature pipeline: lags, rolling stats, calendar, price, event flags.

Every y-derived feature is computed from y shifted by at least one day within its own series, so
a feature value at date d never depends on y at d or later — see tests/test_features.py for the
leakage test (perturbing future y and confirming features at an earlier cutoff don't move).
Calendar/price/category columns pass straight through unshifted: those are known in advance at
forecast time (you know tomorrow's calendar, and your own pricing plan, today), so using the
current row's own value there is not leakage.
"""

from __future__ import annotations

import pandas as pd

from reorderpoint.grain import GRAIN

# Lags and windows in periods of the active grain: (1, 7, 14, 28) / (7, 28) / 7 days for Track A,
# (1, 2, 4, 13) / (4, 13) / 4 weeks for Track B (reorderpoint/grain.py).
LAGS = GRAIN.lags
ROLLING_WINDOWS = GRAIN.rolling_windows
PRICE_CHANGE_LAG = GRAIN.price_change_lag
PRICE_CHANGE_COLUMN = f"price_change_{PRICE_CHANGE_LAG}"  # price_change_7 on Track A

CATEGORICAL_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]
PASSTHROUGH_NUMERIC_COLUMNS = ["price", "wday", "month", "year", "snap", "weekofyear"]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["series_id", "date"]).reset_index(drop=True)

    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby("series_id", sort=False)["y"].shift(lag)

    df["_shifted_y"] = df.groupby("series_id", sort=False)["y"].shift(1)
    grp_shifted = df.groupby("series_id", sort=False)["_shifted_y"]
    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = grp_shifted.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
        df[f"roll_std_{window}"] = grp_shifted.transform(
            lambda s, w=window: s.rolling(w, min_periods=2).std()
        )
        df[f"roll_zero_rate_{window}"] = grp_shifted.transform(
            lambda s, w=window: (s == 0).rolling(w, min_periods=1).mean()
        )
    df = df.drop(columns=["_shifted_y"])

    if "price" in df.columns:
        df[PRICE_CHANGE_COLUMN] = df.groupby("series_id", sort=False)["price"].transform(
            lambda s: s / s.shift(PRICE_CHANGE_LAG) - 1
        )

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Engineered + passthrough columns actually present — Track B has far fewer than Track A."""
    generated = (
        [f"lag_{lag}" for lag in LAGS]
        + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
        + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
        + [f"roll_zero_rate_{w}" for w in ROLLING_WINDOWS]
        + ([PRICE_CHANGE_COLUMN] if "price" in df.columns else [])
    )
    passthrough = [c for c in PASSTHROUGH_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if c in df.columns]
    return generated + passthrough


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_COLUMNS if c in df.columns]
