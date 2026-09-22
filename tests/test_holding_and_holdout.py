"""Holding-cost re-pricing and the held-out target selection. Synthetic runs with known answers:
the point is that the machinery detects a ranking flip and an unstable optimum when one exists."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reorderpoint import holding_sensitivity as hs
from reorderpoint import holdout_target as ho

H0 = 0.02


def _detail(spec: dict[str, tuple[float, float]], n_series: int = 30) -> pd.DataFrame:
    """spec: model -> (holding, stockout) per series per fold at the base rate."""
    rows = []
    for model, (hold, stock) in spec.items():
        for sid in range(n_series):
            for fold in (1, 2):
                rows.append(
                    {
                        "policy": "csl",
                        "model": model,
                        "series_id": f"S{sid}",
                        "fold": fold,
                        "holding_cost": hold,
                        "stockout_cost": stock,
                        "total_cost": hold + stock,
                    }
                )
    return pd.DataFrame(rows)


def test_reprice_is_exact_at_the_base_rate_and_linear_in_holding():
    d = _detail({"A": (100.0, 10.0)})
    assert hs.reprice(d, H0, H0)["total_cost"].tolist() == d["total_cost"].tolist()
    cheap = hs.reprice(d, H0 / 10, H0)
    assert cheap["total_cost"].iloc[0] == pytest.approx(10.0 + 10.0)  # holding/10 + stockout


def test_identity_check_rejects_components_that_do_not_sum_to_the_total():
    d = _detail({"A": (100.0, 10.0)}).assign(total_cost=999.0)
    with pytest.raises(ValueError):
        hs.check_reprice_identity(d, H0)


def test_ranking_flips_when_holding_gets_cheap_and_is_reported_unstable():
    """A lean model (low holding, high stockout) wins at the configured rate; a well-stocked one
    wins once holding is nearly free. The exact shape of the project's finding."""
    d = _detail({"Lean": (50.0, 50.0), "Stocked": (100.0, 10.0)})
    ranks = hs.model_costs(d, H0)
    first = {label: g.sort_values("rank").iloc[0]["model"] for label, g in ranks.groupby("rate")}
    assert first[hs.CONFIGURED] == "Lean"
    assert first[hs.ANNUAL] == "Stocked"
    assert not hs.ranking_is_stable(ranks)


def test_ranking_is_stable_when_one_model_dominates_both_components():
    d = _detail({"Good": (50.0, 10.0), "Bad": (100.0, 60.0)})
    assert hs.ranking_is_stable(hs.model_costs(d, H0))


def _sweep_rows(cell=("normal", "intermittency")) -> pd.DataFrame:
    targets = [0.8, 0.9, 0.95, 0.99]
    holding = [50.0, 70.0, 90.0, 150.0]
    stockout = [200.0, 120.0, 90.0, 60.0]
    csl = [0.6, 0.7, 0.78, 0.9]
    rows = []
    for fold in (1, 2, 3):
        for t, h, s, c in zip(targets, holding, stockout, csl, strict=True):
            rows.append(
                {
                    "form": cell[0],
                    "granularity": cell[1],
                    "target": t,
                    "fold": fold,
                    "model": "m",
                    "holding_cost": h,
                    "stockout_cost": s,
                    "total_cost": h + s,
                    "cycle_service_level": c,
                    "fill_rate": c,
                }
            )
    return pd.DataFrame(rows)


def test_optimal_target_moves_up_and_hits_the_grid_edge_as_holding_gets_cheap():
    out = hs.optimal_targets(_sweep_rows(), H0).set_index("rate")
    assert out.loc[hs.CONFIGURED, "argmin_target"] == 0.95
    assert not out.loc[hs.CONFIGURED, "at_grid_edge"]
    assert out.loc[hs.ANNUAL, "argmin_target"] == 0.99
    assert out.loc[hs.ANNUAL, "at_grid_edge"]  # the answer is a clipped one, and says so


def test_selection_can_be_restricted_to_training_folds_only():
    rows = _sweep_rows()
    # make fold 3 alone prefer a different target; selecting on folds 1-2 must ignore it
    rows.loc[(rows["fold"] == 3) & (rows["target"] == 0.8), "stockout_cost"] = 0.0
    rows["total_cost"] = rows["holding_cost"] + rows["stockout_cost"]
    train = hs.optimal_targets(rows, H0, folds=[1, 2]).set_index("rate")
    assert train.loc[hs.CONFIGURED, "argmin_target"] == 0.95


def _matched_rows() -> pd.DataFrame:
    base = ("normal", "intermittency")
    fine = ("normal", "volume_quintile")
    rows = []
    for fold in (1, 2, 3, 4):
        rows.append(
            {
                "form": base[0],
                "granularity": base[1],
                "target": 0.95,
                "fold": fold,
                "cycle_service_level": 0.80,
            }
        )
        for t, c in [(0.95, 0.77), (0.96, 0.79), (0.97, 0.81), (0.98, 0.83)]:
            rows.append(
                {
                    "form": fine[0],
                    "granularity": fine[1],
                    "target": t,
                    "fold": fold,
                    "cycle_service_level": c,
                }
            )
    return pd.DataFrame(rows)


def test_matched_target_is_the_smallest_that_reaches_baseline_service():
    assert ho.matched_target(_matched_rows(), [1, 2, 3]) == 0.97


def test_matched_target_is_none_when_no_grid_point_reaches_baseline():
    rows = _matched_rows().assign(cycle_service_level=lambda d: d["cycle_service_level"] * 0 + 0.5)
    rows.loc[rows["granularity"] == "intermittency", "cycle_service_level"] = 0.9
    assert ho.matched_target(rows, [1, 2, 3]) is None


def _curve(costs: dict[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"form": "normal", "granularity": "volume_quintile", "target": t, "cost": c}
            for t, c in costs.items()
        ]
    )


def test_stability_passes_when_the_held_out_optimum_is_close():
    out = ho.stability(0.97, _curve({0.95: 105.0, 0.96: 100.0, 0.97: 101.0, 0.99: 130.0}), ho.FINE)
    assert out["held_opt"] == 0.96 and out["stable"]


def test_stability_fails_when_the_optimum_is_far_and_the_regret_is_large():
    out = ho.stability(0.95, _curve({0.95: 120.0, 0.97: 110.0, 0.995: 100.0}), ho.FINE)
    assert out["held_opt"] == 0.995
    assert out["regret"] == pytest.approx(0.20)
    assert not out["stable"]


def test_stability_passes_when_far_but_flat():
    """A curve flat enough that the selected target costs <1% more is not a fold-specific value
    worth rejecting, even if the argmin itself sits far away."""
    out = ho.stability(0.95, _curve({0.95: 100.5, 0.99: 100.0}), ho.FINE)
    assert out["stable"]


def test_walk_forward_selects_only_on_earlier_folds():
    rows = _matched_rows().assign(
        total_cost=100.0, holding_cost=50.0, stockout_cost=50.0, fill_rate=0.8, model="m"
    )
    wf = ho.walk_forward(rows, H0, H0)
    assert wf["scored_fold"].tolist() == [2, 3, 4]
    assert (wf["selected_target"] == 0.97).all()
    assert np.isfinite(wf["cost_diff"]).all()
