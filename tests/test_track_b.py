"""The Track B verdicts, on synthetic cost frames whose right answers are known.

`reorderpoint/track_b.py` turns simulation output into the four verdicts `docs/track_b.md`
registered. These tests build the frames by hand so each rule can be seen to say yes when it should
and no when it should not. No model is fit.
"""

import numpy as np
import pandas as pd
import pytest

from reorderpoint import track_b as tb

N_SERIES = 40


def _detail(costs: dict[str, float], noise: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """One row per (model, fold, series); each model's per-series cost centred on costs[model]."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 1.5, N_SERIES)  # series differ in size, shared by every model
    rows = []
    for model, cost in costs.items():
        for fold in (1, 2, 3, 4):
            for i in range(N_SERIES):
                jitter = 1 + noise * rng.standard_normal()
                total = cost * base[i] * jitter / (base.sum() * 4) * 4
                rows.append(
                    {"model": model, "fold": fold, "series_id": f"s{i}", "total_cost": total}
                )
    return pd.DataFrame(rows)


def _summary(costs: dict[str, float]) -> dict:
    return {m: {"cost": c} for m, c in costs.items()}


LADDER = {m: 100.0 for m in tb.LADDER}


def test_finding_1_replicates_when_the_policy_saves_far_more_than_the_models_differ():
    lot0 = {m: 1000.0 for m in tb.LADDER}
    lot2 = {m: 300.0 + i for i, m in enumerate(tb.LADDER)}  # models differ by a few dollars
    out = tb.finding_1(_detail(lot0), _detail(lot2, seed=1), _summary(lot2))
    assert out["all_exclude_zero"] and out["mean_saving"] > 600 and out["spread"] < 10
    assert out["replicated"]


def test_finding_1_fails_when_the_models_differ_by_more_than_the_policy_saves():
    lot0 = {m: 1000.0 for m in tb.LADDER}
    lot2 = {m: 950.0 for m in tb.LADDER}
    lot2["LightGBM"] = 400.0  # one model is cheap enough that the spread dwarfs the saving
    lot2["TSB"] = 1300.0
    out = tb.finding_1(_detail(lot0), _detail(lot2, seed=1), _summary(lot2))
    assert out["spread"] > out["mean_saving"]
    assert not out["replicated"]


def test_finding_1_fails_when_any_model_does_not_clearly_save():
    lot0 = {m: 1000.0 for m in tb.LADDER}
    lot2 = {m: 500.0 for m in tb.LADDER}
    lot2["SeasonalNaive"] = 1000.0  # no saving for one model of the eight
    out = tb.finding_1(_detail(lot0), _detail(lot2, seed=1), _summary(lot2))
    assert not out["all_exclude_zero"] and not out["replicated"]


def test_findings_3_and_4_replicate_when_every_cluster_model_costs_the_same():
    same = {m: 500.0 for m in tb.LADDER}
    f3, f4 = tb.findings_3_4(_detail(same, noise=0.02))
    assert f3["n_pairs"] == 21 and not f3["excluding_zero"] and f3["replicated"]
    assert f4["replicated"]
    assert set(f3["vs_lightgbm"]) == set(tb.CLUSTER) - {"LightGBM"}


def test_finding_3_names_the_pair_that_is_not_tied_and_finding_4_can_still_hold():
    costs = {m: 500.0 for m in tb.LADDER}
    costs["TSB"] = 300.0  # clearly cheaper than everything else
    f3, f4 = tb.findings_3_4(_detail(costs, noise=0.02))
    assert not f3["replicated"]
    assert any("TSB" in pair for pair in f3["excluding_zero"])
    assert len(f3["excluding_zero"]) == 6  # TSB against each of the other six
    assert f4["replicated"]  # MovingAverage against LightGBM is untouched by TSB


def test_finding_4_fails_when_the_trailing_mean_is_clearly_dearer_than_lightgbm():
    costs = {m: 500.0 for m in tb.LADDER}
    costs["MovingAverage"] = 800.0
    _, f4 = tb.findings_3_4(_detail(costs, noise=0.02))
    assert not f4["replicated"]
    assert f4["moving_average_vs_lightgbm"]["lo"] > 0


def test_the_thresholds_are_the_registered_ones():
    assert (tb.COVERAGE_TOLERANCE, tb.CSL_GAP) == (0.03, 0.03)
    assert tb.UNDERSHOOT_MIN == pytest.approx(1 / 3)
    assert tb.MARGINS == (0.20, 0.275, 0.40, 0.50) and tb.LEADS == (1, 2, 4)
    assert (tb.DEFAULT_MARGIN, tb.DEFAULT_LEAD, tb.TARGET) == (0.275, 2, 0.95)
    assert len(tb.CLUSTER) == 7 and "SeasonalNaive" not in tb.CLUSTER


def _results(f1: bool, f2: bool, f3: bool, f4: bool, flip_at: tuple | None = None) -> dict:
    cells = {}
    for m in tb.MARGINS:
        for lead in tb.LEADS:
            flipped = (m, lead) == flip_at
            cells[tb.cell_key(m, lead)] = {
                "finding_1": {"replicated": f1 != flipped},
                "finding_3": {"replicated": f3},
                "finding_4": {"replicated": f4},
            }
    return {"cells": cells, "finding_2": {f"L{lead}": {"replicated": f2} for lead in tb.LEADS}}


def test_verdicts_are_read_from_the_default_cell_and_the_grid_count_is_of_twelve():
    res = _results(True, False, True, False, flip_at=(0.5, 4))
    assert tb.verdicts(res) == {1: True, 2: False, 3: True, 4: False}
    assert tb.holds_across_grid(res, 1) == (11, 12)  # one of the twelve cells flips finding 1
    assert tb.holds_across_grid(res, 3) == (12, 12)


def test_the_verdict_word_and_money_formatting():
    assert tb._word(True) == "replicated" and tb._word(False) == "not replicated"
    assert tb._money(-436.4) == "-$436" and tb._money(1234.6) == "$1,235"
