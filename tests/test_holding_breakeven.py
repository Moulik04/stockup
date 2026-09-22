"""The holding-cost breakeven: closed form vs. an independent root-find, the stamp that stops the
config default relabelling old runs, and the premise that the simulation ignores the rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reorderpoint import backtest as bt
from reorderpoint import decision as dec
from reorderpoint import holding_breakeven as hb
from reorderpoint.config import (
    DEFAULT_ANNUAL_HOLDING_RATE,
    CostParams,
    annual_to_daily,
    daily_to_annual,
    describe_holding_rate,
    load_config,
)

H0 = 0.02


def _detail(spec: dict[str, tuple[float, float]], n_series: int = 35, rate: float | None = None):
    """spec: model -> (holding, stockout) per series per fold, simulated at `H0` unless stamped.

    35 series so the +-spread (period 5 for holding, 7 for stockout) averages out exactly."""
    rows = []
    for model, (hold, stock) in spec.items():
        for sid in range(n_series):
            for fold in (1, 2, 3):
                rows.append(
                    {
                        "policy": "csl",
                        "model": model,
                        "series_id": f"S{sid}",
                        "fold": fold,
                        # series-level spread so a bootstrap has something to resample
                        "holding_cost": hold * (1 + 0.1 * ((sid % 5) - 2)),
                        "stockout_cost": stock * (1 + 0.1 * ((sid % 7) - 3)),
                    }
                )
    df = pd.DataFrame(rows)
    if rate is not None:
        df["holding_cost_rate"] = rate
    return df


# A holds less and stocks out more; B the reverse. Per fold at H0 over 35 series:
# A: holding 3500, stockout 350.  B: holding 7000, stockout 175.
SPEC = {"A": (100.0, 10.0), "B": (200.0, 5.0)}


def test_annual_daily_round_trip_and_the_default():
    assert daily_to_annual(annual_to_daily(0.25)) == pytest.approx(0.25)
    assert load_config().costs.holding_cost_rate == pytest.approx(
        annual_to_daily(DEFAULT_ANNUAL_HOLDING_RATE)
    )
    assert describe_holding_rate(0.02) == "2.0000%/day (730.0%/yr)"


def test_simulated_rate_falls_back_to_the_legacy_value_only_when_unstamped():
    assert hb.simulated_rate(_detail(SPEC)) == hb.LEGACY_SIMULATED_RATE
    assert hb.simulated_rate(_detail(SPEC, rate=0.0007)) == 0.0007
    mixed = pd.concat([_detail(SPEC, rate=0.0007), _detail(SPEC, rate=0.02)])
    with pytest.raises(ValueError, match="different holding rates"):
        hb.simulated_rate(mixed)


def test_breakeven_closed_form_matches_hand_calculation():
    comp = hb.model_components(_detail(SPEC))
    # cost_A(r) = (3500/0.02) r + 350 = 175000 r + 350; cost_B(r) = 350000 r + 175
    # equal when 175 = 175000 r  ->  r = 0.001/day
    assert comp.loc["A", "holding_per_rate"] == pytest.approx(175_000)
    assert comp.loc["B", "stockout"] == pytest.approx(175)
    assert hb.breakeven(comp, "A", "B") == pytest.approx(0.001)
    at = hb.cost_at(comp, 0.001)
    assert at["A"] == pytest.approx(at["B"])
    # below the crossover holding is cheap, so the model that stocks out less (B) wins
    assert hb.cheaper_below(comp, "A", "B") == "B"
    assert hb.cost_at(comp, 0.0001)["B"] < hb.cost_at(comp, 0.0001)["A"]
    assert hb.cost_at(comp, 0.01)["A"] < hb.cost_at(comp, 0.01)["B"]


def test_no_crossover_when_one_model_dominates():
    comp = hb.model_components(_detail({"A": (100.0, 5.0), "B": (200.0, 10.0)}))
    assert hb.breakeven(comp, "A", "B") is None
    row = hb.pairwise_breakevens(comp, ref="A").iloc[0]
    assert row["always_cheaper"] == "A"


def test_numeric_root_agrees_with_the_closed_form():
    d = _detail(SPEC)
    comp = hb.model_components(d)
    assert hb.solve_numerically(d, "A", "B") == pytest.approx(
        hb.breakeven(comp, "A", "B"), rel=1e-9
    )


def test_config_default_cannot_relabel_a_stamped_run():
    """The same simulation, stamped at a different rate with holding scaled to match, gives the
    same crossover — the breakeven follows the stamp, not whatever the config now says."""
    at_h0 = hb.breakeven(hb.model_components(_detail(SPEC)), "A", "B")
    scaled = _detail(SPEC, rate=0.001)
    scaled["holding_cost"] *= 0.001 / H0
    assert hb.breakeven(hb.model_components(scaled), "A", "B") == pytest.approx(at_h0)


def test_bootstrap_interval_brackets_the_point_and_reports_the_plausible_range():
    bb = hb.bootstrap_breakeven(_detail(SPEC), "A", "B", annual_rates=(0.15, 0.30), n_boot=400)
    assert bb.point == pytest.approx(daily_to_annual(0.001))
    assert bb.ci_lo < bb.point < bb.ci_hi
    # 0.001/day is 36.5%/yr: at 15% and 30%/yr, A is cheaper in almost no resample
    assert bb.a_cheaper_at[0.15] < 0.05 and bb.a_cheaper_at[0.30] < 0.05


def test_penalty_needed_for_a_crossover_scales_linearly():
    comp = hb.model_components(_detail(SPEC))
    # crossover is 0.001/day; to move it to 0.0005/day the stockout penalty must halve
    needed = hb.penalty_for_breakeven(comp, daily_to_annual(0.0005), 5.0, "A", "B")
    assert needed == pytest.approx(2.5)


def test_simulation_trajectory_is_invariant_to_the_holding_rate():
    """The premise behind every re-pricing: changing the rate changes what holding costs, not what
    the policy does — same stockouts, same shipments, holding exactly proportional."""
    demand = np.array([3.0, 0.0, 5.0, 2.0, 7.0, 1.0, 0.0, 4.0, 6.0, 2.0] * 3)

    def run(rate: float):
        costs = CostParams(rate, 5.0, 3, 0.95)
        return dec.simulate_series(
            demand, s=8.0, S=8.0, on_hand_start=8.0, lead_time_days=3, unit_cost=2.5, costs=costs
        )

    a, b = run(0.02), run(annual_to_daily(0.25))
    assert a.orders > 0  # the reorder branch is exercised, not just a static stock
    for field in ("stockout_cost", "units_shipped", "orders", "avg_on_hand", "cycles"):
        assert getattr(a, field) == pytest.approx(getattr(b, field))
    assert b.holding_cost == pytest.approx(a.holding_cost * annual_to_daily(0.25) / 0.02)


def test_simulate_policy_stamps_the_rate_it_ran_at():
    rate = 0.0007
    costs = CostParams(rate, 5.0, 3, 0.95)
    decisions = pd.DataFrame(
        {"reorder_point": [8.0], "std": [2.0], "safety_stock": [3.0]}, index=["S0"]
    )
    day = pd.Timestamp("2024-01-01")
    fold = bt.Fold(1, day, day + pd.Timedelta(days=1), day + pd.Timedelta(days=10))
    _, detail = dec._simulate_policy(
        decisions,
        {"S0": np.array([3.0, 0.0, 5.0, 2.0, 7.0, 1.0, 0.0, 4.0, 6.0, 2.0])},
        pd.Series({"S0": 2.5}),
        2.5,
        costs,
        fold,
        "M",
        "csl",
    )
    assert hb.simulated_rate(detail) == rate


def test_chart_renders(tmp_path):
    comp = hb.model_components(_detail(SPEC))
    out = tmp_path / "c.png"
    hb.plot_cost_vs_rate(comp, out, "A", "B", x_max_pct=200)
    assert out.stat().st_size > 5_000
