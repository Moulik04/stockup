"""Paired bootstrap over series (v1.1 Task 3): put a confidence interval on the *difference*
between two conditions (two models, or two safety-stock calibration cells) rather than trusting
the point estimate alone.

The unit of resampling is the series, not the row. Every comparison in this project is run on
the same ~400-series panel and the same folds for both sides of a comparison, so for a given
metric we first collapse each series down to one number per side (summed or pooled across that
series' folds — see the two aggregation regimes below), then resample *series* with replacement.
That is what makes it a *paired* bootstrap: a resample draws the same series (and therefore the
same folds) for both sides at once, so whatever is common to a series or a fold cancels in the
difference, and only genuine between-series variability drives the width of the CI.

Two aggregation regimes come up:

- **Additive** (e.g. total cost): each series contributes a value per fold that gets *summed*
  across series within a fold, then averaged across folds — exactly how `total_cost` is reported
  today (`decision._simulate_policy`, `ablate_safety_stock._cell_summary`). `bootstrap_diff`
  reproduces that by resampling rows of a (series x fold[/model]) matrix and re-summing.
- **Pooled ratio** (fill rate, cycle service level): each series contributes a numerator and a
  denominator (units shipped/demanded, or stockout cycles/total cycles) that get summed within a
  fold *before* dividing, then the ratio is averaged across folds. `bootstrap_ratio_diff` mirrors
  that by resampling numerator/denominator matrices together and re-deriving the ratio.

Both return the same shape of result: point estimate of A - B, a 95% CI, and the fraction of
resamples where the sign disagrees with the point estimate (a cheap, distribution-free read on
"could this be zero" that doesn't require the CI to be symmetric).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 0


def _matrix(
    detail: pd.DataFrame,
    value_col: str,
    series_col: str,
    group_cols: list[str],
    series_ids: np.ndarray,
    group_keys: list,
) -> np.ndarray:
    """(series x group) matrix of `value_col` summed within each (series, group) cell, reindexed
    to a fixed series/group universe so two details can be compared cell-for-cell. Missing
    combinations (a series absent from a fold, e.g. no test-period demand) fill with 0 — the same
    as that series contributing nothing to that fold's pooled sum, which is what actually happens
    downstream today.
    """
    pivot = detail.pivot_table(
        index=series_col, columns=group_cols, values=value_col, aggfunc="sum", fill_value=0.0
    )
    # pivot_table collapses a single-element `columns` list to a flat Index rather than a
    # length-1 MultiIndex, while `group_keys` (built via itertuples) is always a list of tuples
    # — normalise both to a MultiIndex so `reindex` actually matches columns instead of silently
    # dropping every one of them to the fill value.
    if not isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = pd.MultiIndex.from_arrays([pivot.columns])
    pivot = pivot.reindex(
        index=series_ids, columns=pd.MultiIndex.from_tuples(group_keys), fill_value=0.0
    )
    return pivot.to_numpy(dtype=float)


def _group_keys(detail_a: pd.DataFrame, detail_b: pd.DataFrame, group_cols: list[str]) -> list:
    keys_a = set(detail_a[group_cols].drop_duplicates().itertuples(index=False, name=None))
    keys_b = set(detail_b[group_cols].drop_duplicates().itertuples(index=False, name=None))
    return sorted(keys_a | keys_b)


def bootstrap_diff(
    detail_a: pd.DataFrame,
    detail_b: pd.DataFrame,
    value_col: str,
    series_col: str = "series_id",
    group_cols: list[str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:
    """CI on mean_group(sum_series(A)) - mean_group(sum_series(B)) under series resampling.

    `group_cols` defaults to `["fold"]`. Passing `["fold", "model"]` reproduces a report metric
    averaged across folds *and* models in one call (e.g. the ablation cells' headline cost,
    which is meaned over both) — the resample still draws once per series, shared across every
    (fold, model) column, so it stays paired.
    """
    group_cols = group_cols or ["fold"]
    series_ids = np.array(
        sorted(set(detail_a[series_col].unique()) | set(detail_b[series_col].unique()))
    )
    group_keys = _group_keys(detail_a, detail_b, group_cols)
    n = len(series_ids)

    mat_a = _matrix(detail_a, value_col, series_col, group_cols, series_ids, group_keys)
    mat_b = _matrix(detail_b, value_col, series_col, group_cols, series_ids, group_keys)

    point = float(mat_a.sum(axis=0).mean() - mat_b.sum(axis=0).mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_a = mat_a[idx].sum(axis=1).mean(axis=1)  # (n_boot,)
    boot_b = mat_b[idx].sum(axis=1).mean(axis=1)
    boot_diff = boot_a - boot_b

    lo, hi = (float(v) for v in np.quantile(boot_diff, [alpha / 2, 1 - alpha / 2]))
    sign = np.sign(point) if point != 0 else 0.0
    frac_sign_flip = float(np.mean(np.sign(boot_diff) != sign)) if sign != 0 else float("nan")
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "crosses_zero": bool(lo <= 0 <= hi),
        "frac_sign_flip": frac_sign_flip,
        "n_series": n,
    }


def bootstrap_ratio_diff(
    detail_a: pd.DataFrame,
    detail_b: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    series_col: str = "series_id",
    group_cols: list[str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:
    """Same as `bootstrap_diff`, but for a pooled ratio (fill rate: units_shipped/units_demanded;
    CSL: 1 - stockout_cycles/cycles — pass `numerator_col="cycles_ok"` after precomputing
    `cycles - stockout_cycles`, so the ratio itself is "share OK" in both cases).

    Numerator and denominator are summed within each (series-resample x group) cell *before*
    dividing — pooling over units/cycles, not averaging a per-series ratio — matching how
    `fill_rate` and `cycle_service_level` are actually computed in the report today.
    """
    group_cols = group_cols or ["fold"]
    series_ids = np.array(
        sorted(set(detail_a[series_col].unique()) | set(detail_b[series_col].unique()))
    )
    group_keys = _group_keys(detail_a, detail_b, group_cols)
    n = len(series_ids)

    num_a = _matrix(detail_a, numerator_col, series_col, group_cols, series_ids, group_keys)
    den_a = _matrix(detail_a, denominator_col, series_col, group_cols, series_ids, group_keys)
    num_b = _matrix(detail_b, numerator_col, series_col, group_cols, series_ids, group_keys)
    den_b = _matrix(detail_b, denominator_col, series_col, group_cols, series_ids, group_keys)

    def _pooled_ratio_mean(num: np.ndarray, den: np.ndarray) -> float:
        den_sum = den.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(den_sum > 0, num.sum(axis=0) / den_sum, np.nan)
        return float(np.nanmean(ratio))

    point = _pooled_ratio_mean(num_a, den_a) - _pooled_ratio_mean(num_b, den_b)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    def _boot_stat(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        boot_num = num[idx].sum(axis=1)  # (n_boot, n_groups)
        boot_den = den[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(boot_den > 0, boot_num / boot_den, np.nan)
        return np.nanmean(ratio, axis=1)  # (n_boot,)

    boot_diff = _boot_stat(num_a, den_a) - _boot_stat(num_b, den_b)

    lo, hi = (float(v) for v in np.quantile(boot_diff, [alpha / 2, 1 - alpha / 2]))
    sign = np.sign(point) if point != 0 else 0.0
    frac_sign_flip = float(np.mean(np.sign(boot_diff) != sign)) if sign != 0 else float("nan")
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "crosses_zero": bool(lo <= 0 <= hi),
        "frac_sign_flip": frac_sign_flip,
        "n_series": n,
    }
