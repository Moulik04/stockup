"""Tests for the CI-producing modules built on `bootstrap.py`: the calibration-cell comparisons,
the cost-optimal service level maths, model-vs-model CIs and reconciliation CIs. Synthetic
fixtures throughout, so each expected answer is known by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reorderpoint import ablate_safety_stock as ab
from reorderpoint import model_ci as mc
from reorderpoint import reconcile as rc
from reorderpoint import reconcile_ci as rci
from reorderpoint import service_level_sweep as sw
from reorderpoint.config import CostParams

COSTS = CostParams(
    holding_cost_rate=0.02,
    stockout_penalty_per_unit=5.0,
    lead_time_days=7,
    service_level_target=0.95,
)


# --- critical ratio / cycle length -------------------------------------------------------------


def test_critical_ratio_matches_hand_calculation():
    # Co = 0.02 * $5 * 10 days = $1; CR = 5 / (5 + 1)
    assert sw.critical_ratio(COSTS, 5.0, 10) == pytest.approx(5 / 6)


def test_critical_ratio_falls_with_price_and_with_cycle_length():
    cheap, dear = sw.critical_ratio(COSTS, np.array([1.0, 30.0]), 9.0)
    assert cheap > dear
    assert sw.critical_ratio(COSTS, 5.0, 14) < sw.critical_ratio(COSTS, 5.0, 7)


def test_realised_cycle_days_is_series_days_per_order():
    detail = pd.DataFrame({"orders": [2, 2]})
    # 2 series x 28 days = 56 series-days over 4 orders
    assert sw.realised_cycle_days(detail, 28) == pytest.approx(14.0)


# --- calibration-cell bootstrap ----------------------------------------------------------------


def _cell_detail(form, gran, cost, stockout_cycles, n_series=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_series):
        for fold in (1, 2):
            for model in ("m1", "m2"):
                rows.append(
                    {
                        "form": form,
                        "granularity": gran,
                        "series_id": f"S{sid}",
                        "fold": fold,
                        "model": model,
                        "total_cost": cost + rng.normal(0, 3),
                        "cycles": 4.0,
                        "stockout_cycles": stockout_cycles,
                        "units_shipped": 8.0,
                        "units_demanded": 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_service_regression_reports_a_real_saving_and_a_real_csl_drop():
    base = _cell_detail("normal", "intermittency", cost=100, stockout_cycles=0.5, seed=1)
    best = _cell_detail("normal", "volume_quintile", cost=90, stockout_cycles=1.0, seed=2)
    detail = pd.concat([base, best], ignore_index=True)

    out = ab.bootstrap_service_regression(
        detail, ("normal", "intermittency"), ("normal", "volume_quintile")
    )
    # cost is best - baseline: a saving is negative; per (fold, model) that is 80 series x -10
    assert out["cost"]["point"] < 0
    assert not out["cost"]["crosses_zero"]
    # CSL = 1 - stockout/cycles: 0.875 (base) vs 0.75 (best), so best - baseline = -0.125
    assert out["csl"]["point"] == pytest.approx(-0.125)
    assert not out["csl"]["crosses_zero"]
    # identical fill rates on both sides -> exactly zero difference
    assert out["fill_rate"]["point"] == pytest.approx(0.0)


def test_cell_comparisons_cover_every_pair_once():
    cells = ab.TWO_BY_TWO_CELLS
    detail = pd.concat(
        [
            _cell_detail(f, g, cost=100 + i, stockout_cycles=0.5, seed=i)
            for i, (f, g) in enumerate(cells)
        ],
        ignore_index=True,
    )
    out = ab.bootstrap_cell_comparisons(detail, cells)
    assert len(out) == len(cells) * (len(cells) - 1) // 2
    assert set(out.columns) >= {"cell_a", "cell_b", "point", "ci_lo", "ci_hi", "crosses_zero"}


def test_form_effect_note_reports_both_granularities_honestly():
    """The two granularities can disagree (coarse: empirical cheaper; fine: not distinguishable).
    Rows are normal - empirical, so a positive difference means empirical is the cheaper form.
    The note must say each, with the right sign, not one blanket verdict."""
    ci = pd.DataFrame(
        [
            {
                "cell_a": "normal×intermittency",
                "cell_b": "empirical×intermittency",
                "point": 186.0,
                "ci_lo": 53.0,
                "ci_hi": 324.0,
                "crosses_zero": False,
            },
            {
                "cell_a": "normal×volume_tercile_within_intermittency",
                "cell_b": "empirical×volume_tercile_within_intermittency",
                "point": -89.0,
                "ci_lo": -181.0,
                "ci_hi": 3.0,
                "crosses_zero": True,
            },
        ]
    )
    note = ab._form_effect_note(ci)
    assert "empirical form is cheaper by $186" in note
    assert "not distinguishable at this sample size" in note
    assert "data-volume" in note
    assert ab._form_effect_note(None) == ""

    # and the sign flips the wording when normal is the cheaper one
    flipped = ci.assign(point=[-186.0, -89.0], ci_lo=[-324.0, -181.0], ci_hi=[-53.0, -10.0])
    flipped["crosses_zero"] = False
    assert "more expensive by $186" in ab._form_effect_note(flipped)


# --- model-vs-model ----------------------------------------------------------------------------


def _model_frames(lgb_cost: float, other_cost: float, n_series=100):
    rng = np.random.default_rng(3)
    cost_rows, mase_rows = [], []
    for model in [mc.FOCUS, *mc.OTHERS]:
        base = lgb_cost if model == mc.FOCUS else other_cost
        for sid in range(n_series):
            for fold in (1, 2):
                cost_rows.append(
                    {
                        "policy": "csl",
                        "model": model,
                        "series_id": f"S{sid}",
                        "fold": fold,
                        "total_cost": base + rng.normal(0, 2),
                        "units_shipped": 8.0,
                        "units_demanded": 10.0,
                        "cycles": 3.0,
                        "stockout_cycles": 1.0,
                    }
                )
                mase_rows.append(
                    {
                        "model": model,
                        "series_id": f"S{sid}",
                        "fold": fold,
                        "mase": 1.5 + rng.normal(0, 0.3),
                    }
                )
    return pd.DataFrame(cost_rows), pd.DataFrame(mase_rows)


def test_model_comparison_finds_a_real_cost_gap_and_a_null_mase_gap():
    cost, mase = _model_frames(lgb_cost=90.0, other_cost=100.0)
    table = mc.build_comparisons(cost, mase)
    cost_rows = table[table["metric"] == "cost per fold ($)"]
    mase_rows = table[table["metric"] == "MASE"]
    assert len(cost_rows) == len(mc.OTHERS)
    assert (~cost_rows["crosses_zero"]).all() and (cost_rows["point"] < 0).all()
    # MASE is drawn from the same distribution for every model: nothing to find
    assert mase_rows["crosses_zero"].all()
    assert table.attrs["n_mase_series"] == 100


def test_model_comparison_report_labels_what_it_cannot_distinguish():
    cost, mase = _model_frames(lgb_cost=100.0, other_cost=100.0)
    report = mc.build_report(mc.build_comparisons(cost, mase))
    assert "not distinguishable at this sample size" in report
    assert "NBEATS" in report  # the uncovered comparison is disclosed, not silently dropped


# --- reconciliation ----------------------------------------------------------------------------


def _node_frame(level: str, n_nodes: int, mintrace_delta: float, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_nodes):
        node_base = 1.0 + rng.normal(0, 0.2)
        for fold in (1, 2):
            for method, delta in (("base", 0.0), ("MinTrace", mintrace_delta), ("BottomUp", 0.0)):
                rows.append(
                    {
                        "series_id": f"{level}-{i}",
                        "level": level,
                        "method": method,
                        "fold": fold,
                        "mase": node_base + delta + rng.normal(0, 0.01),
                        "rmsse": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_levels_with_too_few_nodes_are_not_bootstrapped():
    top, bottom = rc.LEVEL_NAMES[0], rc.LEVEL_NAMES[-1]
    nodes = pd.concat(
        [_node_frame(top, 2, 0.05), _node_frame(bottom, 60, -0.10)], ignore_index=True
    )
    table = rci.level_comparisons(nodes)
    few = table[(table["level"] == top) & (table["method"] == "MinTrace")].iloc[0]
    assert not few["testable"]
    assert np.isnan(few["ci_lo"]) and few["point"] == pytest.approx(0.05, abs=0.02)

    many = table[(table["level"] == bottom) & (table["method"] == "MinTrace")].iloc[0]
    assert many["testable"]
    assert many["point"] == pytest.approx(-0.10, abs=0.01)
    assert not many["crosses_zero"]


def test_a_tiny_reconciliation_effect_inside_the_noise_crosses_zero():
    bottom = rc.LEVEL_NAMES[-1]
    rng = np.random.default_rng(9)
    rows = []
    for i in range(100):
        node_base = 1.0 + rng.normal(0, 0.3)
        for fold in (1, 2):
            base = node_base + rng.normal(0, 0.05)
            for method, off in (("base", 0.0), ("MinTrace", 0.001), ("BottomUp", 0.0)):
                rows.append(
                    {
                        "series_id": f"n{i}",
                        "level": bottom,
                        "method": method,
                        "fold": fold,
                        "mase": base + off + rng.normal(0, 0.05),
                        "rmsse": 1.0,
                    }
                )
    table = rci.level_comparisons(pd.DataFrame(rows))
    row = table[(table["level"] == bottom) & (table["method"] == "MinTrace")].iloc[0]
    assert row["crosses_zero"]
