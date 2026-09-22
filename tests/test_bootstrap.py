"""Paired bootstrap over series: point estimates must reproduce a naive calculation exactly,
and CIs must behave sensibly (bracket the point estimate, narrow with less noise / more series,
detect a real effect, and stay wide/crossing-zero for two draws from the same distribution).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reorderpoint import bootstrap as boot


def _cost_detail(costs: dict[str, dict[int, float]]) -> pd.DataFrame:
    """costs: {series_id: {fold: total_cost}}."""
    rows = [
        {"series_id": sid, "fold": fold, "total_cost": v}
        for sid, per_fold in costs.items()
        for fold, v in per_fold.items()
    ]
    return pd.DataFrame(rows)


def test_bootstrap_diff_point_estimate_matches_naive_calculation():
    # A costs 10/series/fold, B costs 8/series/fold, 3 series x 2 folds each.
    a = _cost_detail({f"S{i}": {1: 10.0, 2: 10.0} for i in range(3)})
    b = _cost_detail({f"S{i}": {1: 8.0, 2: 8.0} for i in range(3)})
    out = boot.bootstrap_diff(a, b, value_col="total_cost", n_boot=200)
    # mean-per-fold sum: fold total = 3*10=30 (A) vs 3*8=24 (B); mean over folds is the same
    # since both folds are identical here -> point diff = 30 - 24 = 6
    assert out["point"] == 6.0
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]
    assert out["n_series"] == 3


def test_bootstrap_diff_ci_narrows_with_more_series():
    rng = np.random.default_rng(0)

    def make(n_series: int, seed: int) -> pd.DataFrame:
        r = np.random.default_rng(seed)
        return _cost_detail(
            {
                f"S{i}": {1: float(r.normal(100, 10)), 2: float(r.normal(100, 10))}
                for i in range(n_series)
            }
        )

    small_a, small_b = make(10, 1), make(10, 2)
    large_a, large_b = make(200, 3), make(200, 4)

    small = boot.bootstrap_diff(small_a, small_b, value_col="total_cost", n_boot=500)
    large = boot.bootstrap_diff(large_a, large_b, value_col="total_cost", n_boot=500)

    # more series -> narrower CI relative to the number of series pooled (the point compares
    # sums of different sizes, so compare CI width per series rather than raw width)
    small_width = (small["ci_hi"] - small["ci_lo"]) / small["n_series"]
    large_width = (large["ci_hi"] - large["ci_lo"]) / large["n_series"]
    assert large_width < small_width
    del rng


def test_bootstrap_diff_detects_a_real_difference():
    rng_a = np.random.default_rng(10)
    rng_b = np.random.default_rng(11)
    a = _cost_detail(
        {f"S{i}": {f: float(rng_a.normal(120, 5)) for f in (1, 2, 3)} for i in range(150)}
    )
    b = _cost_detail(
        {f"S{i}": {f: float(rng_b.normal(100, 5)) for f in (1, 2, 3)} for i in range(150)}
    )
    out = boot.bootstrap_diff(a, b, value_col="total_cost", n_boot=500)
    assert out["point"] > 0
    assert not out["crosses_zero"]
    assert out["frac_sign_flip"] < 0.05


def test_bootstrap_diff_does_not_falsely_reject_identical_distributions():
    rng = np.random.default_rng(20)
    a = _cost_detail(
        {f"S{i}": {f: float(rng.normal(100, 20)) for f in (1, 2, 3)} for i in range(30)}
    )
    b = _cost_detail(
        {f"S{i}": {f: float(rng.normal(100, 20)) for f in (1, 2, 3)} for i in range(30)}
    )
    out = boot.bootstrap_diff(a, b, value_col="total_cost", n_boot=1000)
    assert out["crosses_zero"]


def test_bootstrap_diff_groups_by_fold_and_model_together():
    a = _cost_detail({f"S{i}": {1: 10.0, 2: 10.0} for i in range(4)}).assign(model="m1")
    a2 = _cost_detail({f"S{i}": {1: 20.0, 2: 20.0} for i in range(4)}).assign(model="m2")
    a = pd.concat([a, a2], ignore_index=True)
    b = _cost_detail({f"S{i}": {1: 8.0, 2: 8.0} for i in range(4)}).assign(model="m1")
    b2 = _cost_detail({f"S{i}": {1: 8.0, 2: 8.0} for i in range(4)}).assign(model="m2")
    b = pd.concat([b, b2], ignore_index=True)

    out = boot.bootstrap_diff(
        a, b, value_col="total_cost", group_cols=["fold", "model"], n_boot=100
    )
    # per (fold, model) sums: m1 40 vs 32 (diff 8), m2 80 vs 32 (diff 48) -> mean diff = 28
    assert out["point"] == 28.0


def _ratio_detail(rows: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["series_id", "fold", "shipped", "demanded"])


def test_bootstrap_ratio_diff_point_matches_pooled_fill_rate():
    a = _ratio_detail([("S0", 1, 8.0, 10.0), ("S1", 1, 18.0, 20.0)])
    b = _ratio_detail([("S0", 1, 5.0, 10.0), ("S1", 1, 15.0, 20.0)])
    out = boot.bootstrap_ratio_diff(
        a, b, numerator_col="shipped", denominator_col="demanded", n_boot=200
    )
    # pooled fill rate A = (8+18)/(10+20) = 26/30; B = (5+15)/(10+20) = 20/30
    assert out["point"] == pytest.approx(26 / 30 - 20 / 30)


def test_bootstrap_ratio_diff_ci_brackets_point_and_flags_no_effect():
    rng = np.random.default_rng(5)
    rows_a, rows_b = [], []
    for i in range(100):
        demand = 20.0
        shipped_a = float(np.clip(rng.normal(16, 2), 0, demand))
        shipped_b = float(np.clip(rng.normal(16, 2), 0, demand))
        rows_a.append((f"S{i}", 1, shipped_a, demand))
        rows_b.append((f"S{i}", 1, shipped_b, demand))
    a = _ratio_detail(rows_a)
    b = _ratio_detail(rows_b)
    out = boot.bootstrap_ratio_diff(
        a, b, numerator_col="shipped", denominator_col="demanded", n_boot=500
    )
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]
    assert out["crosses_zero"]
