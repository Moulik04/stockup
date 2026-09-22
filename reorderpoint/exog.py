"""Forward-looking exogenous inputs for serving and calibration.

Real deployments would source these from a promo/pricing/calendar system; this project has none,
so the future is a repeat-the-recent-pattern proxy (disclosed in docs/serving.md). Lives here, not
in `serve.py`, so `calibration.py` (imported by `train.py`, which `serve.py` imports) can use the
same proxy without an import cycle.
"""

from __future__ import annotations

import pandas as pd

REFERENCE_WINDOW_DAYS = 7  # this project's weekly seasonality period (SEASON_LENGTH elsewhere)


def future_exog_from_trailing_window(
    panel: pd.DataFrame, series_ids: list[str], horizon: int
) -> pd.DataFrame:
    """Production stand-in for real forward-looking exog (future calendar/events/price): repeats
    each series' most recent `REFERENCE_WINDOW_DAYS`-day row pattern, tiled to cover `horizon`,
    with dates advanced past the panel's last known date. That reference window is a *fixed*
    size, not sized to `horizon` itself — an earlier version used `group.tail(horizon)`, which
    meant day 1's proxy price/event/SNAP values silently shifted depending on how many days were
    requested (a different historical day became "day 1 of the template" for every horizon
    length). With a fixed window, day 1 of any forecast always means the same thing. Calendar
    fields that are genuinely knowable in advance (wday/month/year) are recomputed from the real
    future dates rather than carried over stale; everything else (price, events, SNAP) is a
    repeat-the-recent-pattern proxy — a real deployment would source these from an actual
    pricing/promo/calendar system, the same kind of disclosed simplification as y-as-demand-proxy
    in docs/data.md.
    """
    sub = panel[panel["series_id"].isin(series_ids)].sort_values(["series_id", "date"])
    last_date = panel["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    frames = []
    for series_id, group in sub.groupby("series_id"):
        reference = group.tail(REFERENCE_WINDOW_DAYS).reset_index(drop=True)
        if reference.empty:
            continue
        reps = horizon // len(reference) + 1
        trailing = pd.concat([reference] * reps, ignore_index=True).iloc[:horizon].copy()
        trailing["date"] = future_dates
        if "wday" in trailing.columns:
            # M5's wday convention is Saturday=1..Friday=7 (confirmed against calendar.csv),
            # not pandas' Monday=0 dayofweek — the model was trained on the former.
            trailing["wday"] = ((trailing["date"].dt.dayofweek + 2) % 7) + 1
        if "month" in trailing.columns:
            trailing["month"] = trailing["date"].dt.month
        if "year" in trailing.columns:
            trailing["year"] = trailing["date"].dt.year
        trailing["series_id"] = series_id
        frames.append(trailing.drop(columns=["y"], errors="ignore"))
    return pd.concat(frames, ignore_index=True)
