"""Safety-stock calibration schemes: distributional form x pooling granularity.

The decision layer sizes safety stock from pooled forecast residuals (docs/decision.md). Two
independent choices are baked into that, and v1.1 Task 5 separates them:

- **Form** — how a bucket of residuals becomes a buffer. `normal` takes `z * std`, assuming a
  symmetric normal lead-time demand. `empirical` takes the residual distribution's own quantile
  at the target service level, assuming nothing about its shape.
- **Granularity** — how series are pooled. `intermittency` is the current two-bucket split
  (>50% zero training days). `volume_tercile_within_intermittency` crosses that with volume
  terciles — named for what it actually does here, not what it was designed to do: on this
  panel, intermittency and volume are near-duplicate variables (every `regular` series is
  already in the top volume tercile), so it only ever subdivides the `intermittent` class three
  ways, not a clean 2x3. It was originally called `intermittency_volume`, a name that implied a
  balanced six-cell cross this scheme never delivers; see `assign_buckets`' docstring and
  `reports/safety_stock_<date>.md` for the measured population counts. `volume_quintile` pools
  by volume alone — five balanced buckets carrying the same information (volume already implies
  intermittency here), with none of the population collapse.

Crossing form with the first two granularities gives a 2x2; `volume_quintile` runs as a fifth
cell (paired with the `normal` form only, since form was already shown not to matter — see the
report) rather than doubling the grid to a 2x3.

**The sample-size constraint is real and is reported, not worked around.** ~400 residuals per
fold split two ways leaves ~200 per bucket, so a 95th percentile rests on ~10 tail observations;
split finer it is fewer still. `bootstrap_quantile_ci` resamples the residuals to put a
confidence interval on the quantile *itself*, so a buffer estimated from too few tail points is
visibly uncertain rather than silently precise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

FORMS = ("normal", "empirical")
GRANULARITIES = ("intermittency", "volume_tercile_within_intermittency", "volume_quintile")
# The scheme every result before 2026-09-20 calls "shipped" — the fixed reference the historical
# comparisons ("finer scheme vs shipped") are made against. It never changes.
LEGACY_SCHEME = ("normal", "intermittency")
# The scheme `train.py` calibrates for serving and the harness sizes with by default. One constant,
# so the two cannot disagree about which scheme is the default. Flipping the default is a change
# here (and `make calibrate`); the historical comparisons stay pinned to LEGACY_SCHEME.
DEFAULT_SCHEME = LEGACY_SCHEME
INTERMITTENCY_THRESHOLD = 0.5
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


def train_stats(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(zero rate, mean daily volume) per series over a training window — the two bucketing
    variables. Used by the backtest harness and by the production calibration alike."""
    grouped = train.groupby("series_id")["y"]
    return grouped.apply(lambda s: (s == 0).mean()), grouped.mean()


def _qcut_ranked(vols: pd.Series, n: int, labels: list[str]) -> pd.Series:
    """qcut on ranks, not raw values: mostly-zero demand produces heavily tied volumes, and
    plain qcut refuses to cut when bin edges collide. Ranking first splits ties arbitrarily but
    evenly, which is the intended "n equal-sized groups" rather than an error.
    """
    try:
        cut = pd.qcut(vols.rank(method="first"), n, labels=labels)
    except ValueError:
        cut = pd.Series([labels[len(labels) // 2]] * len(vols), index=vols.index)
    return pd.Series(cut, index=vols.index).astype(str)


def assign_buckets(
    zero_rates: pd.Series,
    volumes: pd.Series | None = None,
    granularity: str = "intermittency",
    threshold: float = INTERMITTENCY_THRESHOLD,
) -> pd.Series:
    """Per-series bucket label.

    `intermittency` reproduces the existing two-bucket split exactly.
    `volume_tercile_within_intermittency` crosses it with volume terciles computed *within the
    calibration window* (never using the outcome period) — at most six buckets, though on this
    panel only the `intermittent` class actually splits (see the module docstring).
    `volume_quintile` pools by volume alone, five balanced buckets, no intermittency dimension.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")

    if granularity == "volume_quintile":
        if volumes is None:
            raise ValueError("volume_quintile granularity requires `volumes`")
        vols = volumes.reindex(zero_rates.index)
        return _qcut_ranked(vols, 5, ["q1", "q2", "q3", "q4", "q5"]).rename("bucket")

    intermittency = pd.Series(
        np.where(zero_rates > threshold, "intermittent", "regular"), index=zero_rates.index
    )
    if granularity == "intermittency":
        return intermittency.rename("bucket")

    if volumes is None:
        raise ValueError("volume_tercile_within_intermittency granularity requires `volumes`")
    vols = volumes.reindex(zero_rates.index)
    tercile = _qcut_ranked(vols, 3, ["lo", "mid", "hi"])
    return (intermittency + "_" + tercile).rename("bucket")


def bootstrap_quantile_ci(
    residuals: np.ndarray,
    q: float,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) for the `q`-quantile of `residuals`, by resampling with
    replacement. The CI is on the quantile itself — how well this bucket's tail is pinned down
    by the data available — not on any downstream cost.
    """
    clean = residuals[np.isfinite(residuals)]
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    point = float(np.quantile(clean, q))
    if len(clean) == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(n_boot, len(clean)), replace=True)
    boot = np.quantile(draws, q, axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def bucket_safety_stock(
    residuals: pd.Series,
    buckets: pd.Series,
    service_level: float,
    form: str = "normal",
    n_boot: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Per-bucket safety stock, with a bootstrap CI on the underlying statistic.

    `residuals` are `actual - predicted` lead-time demand, so the service-level quantile of the
    residual distribution *is* the buffer that would have covered demand that often — no
    distributional assumption, no z.

    Columns: n, safety_stock, ci_lo, ci_hi, ci_width_frac, std.
    """
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")

    z = norm.ppf(service_level)
    df = pd.DataFrame({"residual": residuals, "bucket": buckets.reindex(residuals.index)})
    rows = []
    for bucket, group in df.groupby("bucket", observed=True):
        values = group["residual"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        std = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
        if form == "normal":
            # CI on z * std, via the bootstrap distribution of the std itself
            point = z * std
            if len(finite) > 1:
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                draws = rng.choice(finite, size=(n_boot, len(finite)), replace=True)
                boot = z * draws.std(axis=1, ddof=1)
                lo, hi = (float(v) for v in np.quantile(boot, [0.025, 0.975]))
            else:
                lo = hi = np.nan
        else:
            point, lo, hi = bootstrap_quantile_ci(finite, service_level, n_boot=n_boot)
        width_frac = (
            (hi - lo) / abs(point) if point and np.isfinite(point) and point != 0 else np.nan
        )
        rows.append(
            {
                "bucket": bucket,
                "n": len(finite),
                "safety_stock": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_width_frac": width_frac,
                "std": std,
            }
        )
    return pd.DataFrame(rows).set_index("bucket")


def per_series_safety_stock(
    residuals: pd.Series,
    zero_rates: pd.Series,
    volumes: pd.Series | None,
    service_level: float,
    form: str,
    granularity: str,
    n_boot: int = N_BOOTSTRAP,
) -> tuple[pd.Series, pd.DataFrame]:
    """Calibrate on one fold's residuals, return (per-series safety stock, per-bucket table).

    A series takes its own bucket's buffer. Buckets with no usable estimate fall back to the
    pooled all-series value, so a thin bucket degrades to the coarser estimate rather than
    dropping the series out of the policy entirely.
    """
    buckets = assign_buckets(zero_rates.reindex(residuals.index), volumes, granularity)
    table = bucket_safety_stock(residuals, buckets, service_level, form=form, n_boot=n_boot)

    pooled = bucket_safety_stock(
        residuals,
        pd.Series("__all__", index=residuals.index),
        service_level,
        form=form,
        n_boot=n_boot,
    )["safety_stock"].iloc[0]

    mapped = buckets.map(table["safety_stock"])
    mapped = mapped.where(mapped.notna(), pooled)
    return mapped.rename("safety_stock"), table


def within_bucket_dispersion(
    per_series_residual_std: pd.Series, buckets: pd.Series
) -> pd.DataFrame:
    """Spread of per-series residual std inside each bucket — the diagnostic that says whether
    pooling in absolute units is defensible. A bucket spanning an order of magnitude is not.
    """
    df = pd.DataFrame(
        {"std": per_series_residual_std, "bucket": buckets.reindex(per_series_residual_std.index)}
    ).dropna()
    out = df.groupby("bucket", observed=True)["std"].agg(
        n="size",
        p10=lambda s: s.quantile(0.10),
        median="median",
        p90=lambda s: s.quantile(0.90),
        max="max",
    )
    out["p90_over_p10"] = out["p90"] / out["p10"].replace(0, np.nan)
    return out
