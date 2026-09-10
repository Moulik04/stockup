"""Realised-error tracking, rolling MASE, and input-drift detection — Phase 6.

`realised_error`/`rolling_mase`/`flag_error_spikes` operate on a *forecast log*: rows of
(series_id, date, y, forecast) accumulated over time as real outcomes arrive (e.g. appended by a
scheduled job comparing each day's `/forecast` output against the actual sales once known — no
such job exists yet, this module is the reusable core it would call). `detect_input_drift`
compares a recent window of raw `y` against each series' training-window baseline, independent of
any forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realised_error(log: pd.DataFrame) -> pd.DataFrame:
    """Adds `error`/`abs_error` columns (`y - forecast`) to a forecast log."""
    out = log.copy()
    out["error"] = out["y"] - out["forecast"]
    out["abs_error"] = out["error"].abs()
    return out


def rolling_mase(errors: pd.DataFrame, scale: float | pd.Series, window: int) -> pd.DataFrame:
    """Per-series trailing rolling MASE: rolling mean absolute error / `scale`.

    `scale` is each series' own in-sample naive-error scale — pass a `pd.Series` indexed by
    series_id (e.g. `backtest._in_sample_scales(train, season_length)["scale_mae"]`) so every
    series is divided by its own scale, not one shared value; a plain `float` applies the same
    scale to every series, which only makes sense for a single-series call (dividing every
    series by one series' scale would defeat MASE's whole point as a scale-free metric).
    """
    out = errors.sort_values(["series_id", "date"]).copy()
    rolling_mae = out.groupby("series_id")["abs_error"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    scale_per_row = out["series_id"].map(scale) if isinstance(scale, pd.Series) else scale
    out["rolling_mase"] = rolling_mae / scale_per_row
    return out


def flag_error_spikes(rolling: pd.DataFrame, threshold: float) -> list[str]:
    """series_id's whose (latest, per input row) rolling MASE exceeds `threshold`."""
    flagged = rolling.loc[rolling["rolling_mase"] > threshold, "series_id"]
    return sorted(flagged.unique().tolist())


def detect_input_drift(
    baseline: pd.DataFrame, recent: pd.DataFrame, z_threshold: float = 3.0
) -> list[str]:
    """Flags series whose recent-window mean `y` has shifted more than `z_threshold` baseline
    standard deviations from its own training-window mean — a simple, explainable drift check
    (not a full distributional test), consistent with this project's "boring tools first" bar.
    """
    baseline_stats = baseline.groupby("series_id")["y"].agg(["mean", "std"])
    recent_mean = recent.groupby("series_id")["y"].mean()

    stats = baseline_stats.join(recent_mean.rename("recent_mean"), how="inner")
    diff = (stats["recent_mean"] - stats["mean"]).abs()

    # A zero-variance baseline (e.g. a series that was exactly 0 throughout, common on this
    # ~77%-zero-sale-day panel) makes any shift categorical, not just statistically significant —
    # flag it directly rather than dividing by zero into an undetectable NaN (NaN > z_threshold
    # is False, so this case was previously never flagged regardless of how large the shift was).
    zero_std = stats["std"] == 0
    z = diff / stats["std"].replace(0, np.nan)
    flagged_normal = z[~zero_std & (z > z_threshold)].index
    flagged_zero_std = diff[zero_std & (diff > 0)].index
    return sorted(set(flagged_normal) | set(flagged_zero_std))
