"""Unit economics and exact re-pricing: a lost sale costs margin x price, stock is valued at cost,
and stored runs re-price per series under any (holding rate, margin) without re-simulating."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reorderpoint import decision as dec
from reorderpoint import economics as econ
from reorderpoint import optimal_target as ot
from reorderpoint import penalty_sensitivity as ps
from reorderpoint.config import (
    DEFAULT_GROSS_MARGIN,
    CostParams,
    annual_to_daily,
    daily_to_annual,
    load_config,
)

H0 = 0.02  # legacy simulated rate


def test_cost_params_unit_economics_and_the_legacy_model():
    econ_costs = CostParams(0.001, 5.0, 7, 0.95, gross_margin=0.25)
    assert econ_costs.stockout_penalty(8.0) == pytest.approx(2.0)  # margin x price
    assert econ_costs.unit_cost(8.0) == pytest.approx(6.0)  # (1 - margin) x price
    legacy = econ_costs.flat()
    assert legacy.gross_margin is None
    assert legacy.stockout_penalty(8.0) == 5.0 and legacy.unit_cost(8.0) == 8.0
    # positional construction (used across the older code and tests) still means legacy
    assert CostParams(0.02, 5.0, 7, 0.95).gross_margin is None


def test_config_defaults_to_unit_economics_and_rejects_ambiguity(monkeypatch):
    monkeypatch.delenv("STOCKOUT_PENALTY_PER_UNIT", raising=False)
    monkeypatch.delenv("GROSS_MARGIN", raising=False)
    assert load_config().costs.gross_margin == DEFAULT_GROSS_MARGIN
    monkeypatch.setenv("STOCKOUT_PENALTY_PER_UNIT", "3.0")  # explicit request for the legacy model
    c = load_config().costs
    assert c.gross_margin is None and c.stockout_penalty_per_unit == 3.0
    monkeypatch.setenv("GROSS_MARGIN", "0.3")
    with pytest.raises(ValueError, match="not both"):
        load_config()


def _sim_costs(margin):
    return CostParams(0.001, 5.0, 3, 0.95, gross_margin=margin)


def _policy_detail(margin: float | None) -> pd.DataFrame:
    """One real `_simulate_policy` run on two series with different prices."""
    from reorderpoint import backtest as bt

    decisions = pd.DataFrame(
        {"reorder_point": [4.0, 4.0], "std": [1.0, 1.0], "safety_stock": [1.0, 1.0]},
        index=["cheap", "dear"],
    )
    demand = np.array([3.0, 0.0, 6.0, 2.0, 7.0, 1.0, 0.0, 4.0, 6.0, 2.0])
    day = pd.Timestamp("2024-01-01")
    fold = bt.Fold(1, day, day + pd.Timedelta(days=1), day + pd.Timedelta(days=10))
    prices = pd.Series({"cheap": 2.0, "dear": 20.0})
    _, detail = dec._simulate_policy(
        decisions,
        {"cheap": demand, "dear": demand},
        prices,
        5.0,
        _sim_costs(margin),
        fold,
        "M",
        "csl",
    )
    return detail.assign(model="M", fold=1, policy="csl")


def test_simulation_prices_each_sku_by_its_own_price_under_unit_economics():
    d = _policy_detail(0.25).set_index("series_id")
    flat = _policy_detail(None).set_index("series_id")
    # same trajectory, so the same lost units and on-hand, different price of each
    lost = flat["stockout_cost"] / 5.0
    assert d.loc["cheap", "stockout_cost"] == pytest.approx(lost["cheap"] * 0.25 * 2.0)
    assert d.loc["dear", "stockout_cost"] == pytest.approx(lost["dear"] * 0.25 * 20.0)
    # holding is valued at cost: (1 - margin) x price instead of price
    assert d.loc["dear", "holding_cost"] == pytest.approx(flat.loc["dear", "holding_cost"] * 0.75)
    assert d["gross_margin"].eq(0.25).all() and d["unit_price"].tolist() == [2.0, 20.0]


def test_reprice_reproduces_a_real_resimulation_exactly():
    """Simulate under legacy pricing, re-price to unit economics, compare with simulating under
    unit economics directly — the premise every analysis here rests on."""
    legacy = _policy_detail(None)
    # simulate at another holding rate too, to exercise both scale factors at once
    annual = 0.4
    direct_costs = CostParams(annual_to_daily(annual), 5.0, 3, 0.95, gross_margin=0.25)
    from reorderpoint import backtest as bt

    decisions = pd.DataFrame(
        {"reorder_point": [4.0, 4.0], "std": [1.0, 1.0], "safety_stock": [1.0, 1.0]},
        index=["cheap", "dear"],
    )
    demand = np.array([3.0, 0.0, 6.0, 2.0, 7.0, 1.0, 0.0, 4.0, 6.0, 2.0])
    day = pd.Timestamp("2024-01-01")
    fold = bt.Fold(1, day, day + pd.Timedelta(days=1), day + pd.Timedelta(days=10))
    _, resim = dec._simulate_policy(
        decisions,
        {"cheap": demand, "dear": demand},
        pd.Series({"cheap": 2.0, "dear": 20.0}),
        5.0,
        direct_costs,
        fold,
        "M",
        "csl",
    )
    rep = econ.reprice(legacy, annual, 0.25).set_index("series_id").sort_index()
    resim = resim.set_index("series_id").sort_index()
    for col in ("holding_cost", "stockout_cost", "total_cost"):
        assert rep[col].to_numpy() == pytest.approx(resim[col].to_numpy(), rel=1e-12)


def test_reprice_at_the_pricing_a_run_used_is_the_identity():
    d = _policy_detail(0.25)
    back = econ.reprice(d, daily_to_annual(0.001), 0.25)
    for col in ("holding_cost", "stockout_cost", "total_cost"):
        assert back[col].to_numpy() == pytest.approx(d[col].to_numpy(), rel=1e-12)


def test_unstamped_legacy_rows_are_read_as_legacy_pricing(monkeypatch):
    """No stamp -> 0.02/day, flat $5, stock at price; the price comes from the unit-price table."""
    monkeypatch.setattr(
        econ,
        "unit_prices",
        lambda panel=None: pd.DataFrame(
            {"fold": [1, 1], "series_id": ["a", "b"], "unit_price": [2.0, 10.0]}
        ),
    )
    d = pd.DataFrame(
        {
            "model": "M",
            "fold": 1,
            "series_id": ["a", "b"],
            "holding_cost": [8.0, 40.0],  # 0.02 x price x on-hand-days: on-hand-days = 200, 200
            "stockout_cost": [5.0, 5.0],  # one lost unit each at the flat $5
        }
    ).assign(total_cost=lambda x: x.holding_cost + x.stockout_cost)
    out = econ.reprice(d, 0.0, 0.25)  # zero holding, 25% margin
    assert out["holding_cost"].tolist() == [0.0, 0.0]
    # lost unit now costs margin x price: 0.5 and 2.5
    assert out["stockout_cost"].tolist() == pytest.approx([0.5, 2.5])
    out2 = econ.reprice(d, daily_to_annual(0.02), None)  # legacy pricing at the legacy rate
    assert out2["total_cost"].tolist() == pytest.approx(d["total_cost"].tolist())


def test_breakeven_under_unit_economics_scales_with_margin_over_one_minus_margin():
    """h*(m) is proportional to m/(1-m): checked on synthetic per-SKU rows where the price
    weights differ between the two models."""
    rows = []
    rng = np.random.default_rng(0)
    for model, hold_scale, lost_scale in (("LightGBM", 1.0, 1.3), ("SeasonalNaive", 1.25, 1.0)):
        for sid in range(30):
            price = float(rng.uniform(2, 12))
            for fold in (1, 2):
                onhand_days = 100 * hold_scale * (0.5 + (sid % 4) / 4)
                lost = 3 * lost_scale * (1 + (sid % 3))
                rows.append(
                    {
                        "policy": "csl",
                        "model": model,
                        "series_id": f"S{sid}",
                        "fold": fold,
                        "unit_price": price,
                        "holding_cost": H0 * price * onhand_days,
                        "stockout_cost": 5.0 * lost,
                    }
                )
    d = pd.DataFrame(rows)
    assert ps.scaling_check(d, margins=[0.1, DEFAULT_GROSS_MARGIN, 0.5, 0.8]) < 1e-9
    assert ps.crossover_pct(d, 0.5) > ps.crossover_pct(d, 0.1)  # a dearer lost sale -> higher


# ---- optimal_target helpers -------------------------------------------------------------------


def test_target_matching_is_exact_not_isclose():
    df = pd.DataFrame({"target": [0.99999, 0.999999, 0.99999], "x": [1, 2, 3]})
    assert ot.at_target(df, 0.99999).sum() == 2 and ot.at_target(df, 0.999999).sum() == 1
    assert np.isclose(0.99999, 0.999999)  # the trap this helper exists to avoid


def test_critical_ratio_under_unit_economics_is_price_free_and_hand_computed():
    c = CostParams(0.001, 5.0, 7, 0.95, gross_margin=0.24)
    # Cu = 0.24 p, Co = 0.001 x 0.76 p x 7  ->  CR = 0.24 / (0.24 + 0.00532)
    cr = ot.critical_ratio(c, np.array([2.0, 9.0]), 7)
    assert cr[0] == pytest.approx(cr[1]) == pytest.approx(0.24 / (0.24 + 0.00532))
    flat = ot.critical_ratio(c.flat(), np.array([2.0, 9.0]), 7)
    assert flat[0] > flat[1]  # flat penalty: a dearer item costs more to hold, so its CR is lower


def test_cycle_outcomes_separates_undershoot_from_sizing_failures():
    # s=5, L=3. Day 0 demand 2 -> on-hand 3 < 5, order placed holding only 3 (an undershoot of 2).
    # Days 1-2 demand 2+2=4 <= s=5, yet on-hand 3 runs out: a stockout the buffer *would* cover.
    undershoot = ot.cycle_outcomes(np.array([2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0]), 5.0, 3)
    assert undershoot[0] == (True, False)
    # s=1, L=2: trigger holding 0.5, then demand 3 > s: a sizing failure
    sizing = ot.cycle_outcomes(np.array([0.5, 3.0, 0.0]), 1.0, 2)
    assert sizing[0] == (True, True)
    # ample stock: cycles open and close with no stockout
    calm = ot.cycle_outcomes(np.array([1.0] * 12), 10.0, 3)
    assert calm and not any(had for had, _ in calm)


def test_nominal_for_csl_and_argmin_on_a_synthetic_curve():
    c = pd.DataFrame(
        {
            "target": [0.9, 0.99, 0.999] * 2,
            "model": ["A"] * 3 + ["B"] * 3,
            "cost": [10.0, 6.0, 7.0, 12.0, 8.0, 9.0],
            "holding": 0.0,
            "stockout": 0.0,
            "csl": [0.80, 0.90, 0.95, 0.80, 0.90, 0.95],
            "fill": 0.9,
        }
    )
    best = ot.argmin_target(c)
    assert best["target"] == 0.99 and not best["at_edge"]
    assert ot.argmin_target(c, "B")["target"] == 0.99
    assert ot.nominal_for_csl(c, 0.9) == 0.99
    assert ot.nominal_for_csl(c, 0.99) is None  # never reached
