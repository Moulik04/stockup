"""The order-up-to policy (S = s + Q): the simulator, the cycle-cause replica it is measured with,
the lot-size decision rule, and the reproducible saturation check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reorderpoint import backtest as bt
from reorderpoint import decision as dec
from reorderpoint import optimal_target as ot
from reorderpoint import order_up_to as oup
from reorderpoint.config import CostParams
from tests.test_serving_calibration import _calib, _costs, _forecasts, _panel

COSTS = CostParams(0.001, 5.0, 3, 0.95)


def _sim(demand, s, S, start):
    return dec.simulate_series(
        demand, s=s, S=S, on_hand_start=start, lead_time_days=3, unit_cost=2.0, costs=COSTS
    )


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("q", [0.0, 1.5, 6.0])
def test_cycle_replica_matches_the_simulator_for_any_order_up_to_level(seed, q):
    """The undershoot number rests on this: the replica's cycle and stockout-cycle counts must
    equal `simulate_series`'s for arbitrary (s, S) and start stock."""
    rng = np.random.default_rng(seed)
    demand = rng.poisson(1.3, 40).astype(float) * (rng.random(40) < 0.7)
    s = float(rng.uniform(2, 9))
    S, start = s + q, s + q / 2
    result = _sim(demand, s, S, start)
    out = ot.cycle_outcomes(demand, s, 3, S, start)
    assert len(out) == result.cycles
    assert sum(had for had, _ in out) == result.stockout_cycles


def test_a_larger_order_up_to_level_removes_undershoot_stockouts():
    """Constant demand of 2/day, lead time 3, s = 7: under S = s an order only replaces the
    deficit while a lead time's demand is still unfilled, so cycles run short; refilling past s
    avoids it."""
    demand = np.full(60, 2.0)
    s = 7.0
    tight = ot.cycle_outcomes(demand, s, 3, s, s)
    loose = ot.cycle_outcomes(demand, s, 3, s + 12.0, s + 6.0)
    stockouts = lambda o: sum(had for had, _ in o)  # noqa: E731
    assert stockouts(tight) > stockouts(loose)
    # neither run has a genuine sizing failure: demand over a lead time (6) never exceeds s
    assert not any(exceeded for _, exceeded in tight + loose)


def test_simulate_policy_lot_multiple_sets_S_and_start_and_zero_is_unchanged():
    decisions = pd.DataFrame(
        {"reorder_point": [8.0], "std": [2.0], "safety_stock": [3.0], "mean": [5.0]}, index=["S0"]
    )
    demand = {"S0": np.array([3.0, 0.0, 5.0, 2.0, 7.0, 1.0, 0.0, 4.0, 6.0, 2.0])}
    day = pd.Timestamp("2024-01-01")
    fold = bt.Fold(1, day, day + pd.Timedelta(days=1), day + pd.Timedelta(days=10))
    args = (decisions, demand, pd.Series({"S0": 2.5}), 2.5, CostParams(0.001, 5.0, 3, 0.95), fold)
    _, base = dec._simulate_policy(*args, "M", "csl")
    _, zero = dec._simulate_policy(*args, "M", "csl", lot_multiple=0.0)
    _, lot = dec._simulate_policy(*args, "M", "csl", lot_multiple=2.0)
    assert base["total_cost"].iloc[0] == zero["total_cost"].iloc[0]
    assert base["order_up_to"].iloc[0] == 8.0 and zero["lot_multiple"].iloc[0] == 0.0
    assert lot["order_up_to"].iloc[0] == pytest.approx(8.0 + 2.0 * 5.0)  # s + m x E[LT demand]
    assert lot["lot_multiple"].iloc[0] == 2.0
    assert lot["orders"].iloc[0] <= base["orders"].iloc[0]  # bigger lots, fewer orders


def _table(costs, cis):
    rows = []
    for m, cost, ci in zip([0.0, 1.0, 2.0], costs, cis, strict=True):
        rows.append(
            {"lot multiple": m, "target": "cost-optimal", "cost / fold": cost, "_cost_ci": ci}
        )
    return pd.DataFrame(rows)


def test_lot_multiple_rule_needs_the_cheapest_policy_to_clear_zero():
    sig = {"point": -50.0, "ci_lo": -90.0, "ci_hi": -10.0, "crosses_zero": False}
    noisy = {"point": -50.0, "ci_lo": -140.0, "ci_hi": 30.0, "crosses_zero": True}
    m, _ = oup.choose_multiple(_table([100.0, 90.0, 80.0], [None, noisy, sig]))
    assert m == 2.0
    m, why = oup.choose_multiple(_table([100.0, 90.0, 80.0], [None, sig, noisy]))
    assert m == 0.0 and "crosses zero" in why  # cheapest is not distinguishable -> keep S = s
    m, _ = oup.choose_multiple(_table([80.0, 90.0, 100.0], [None, sig, sig]))
    assert m == 0.0


def _lot_frame(noisy: bool = False) -> pd.DataFrame:
    rows = []
    series = [f"s{i}" for i in range(5)]
    for fold in [1, 2, 3, 4]:
        for i, sid in enumerate(series):
            for lot in (0.0, 2.0):
                if lot == 0.0:
                    cost = 100.0
                elif not noisy:
                    cost = 80.0 if fold < 4 else 84.0  # cheaper on train and on the held-out fold
                else:
                    cost = 40.0 if i % 2 == 0 else 150.0  # cheaper on average, but not reliably
                rows.append(
                    {
                        "form": oup.LEGACY[0],
                        "granularity": oup.LEGACY[1],
                        "lot_multiple": lot,
                        "target": 0.95,
                        "model": "M",
                        "fold": fold,
                        "series_id": sid,
                        "total_cost": cost,
                        "holding_cost": cost * 0.5,
                        "stockout_cost": cost * 0.5,
                        "cycles": 10,
                        "stockout_cycles": 1,
                        "units_shipped": 90.0,
                        "units_demanded": 100.0,
                        "orders": 5,
                    }
                )
    return pd.DataFrame(rows)


def test_heldout_lot_confirms_a_selection_made_only_on_the_training_folds():
    """`heldout_lot` selects on folds 1-3 only, then scores that choice on fold 4 — a held-out
    check on the lot-multiple decision itself, not just the scheme decision `heldout()` covers."""
    result = oup.heldout_lot(_lot_frame(), multiples=[0.0, 2.0])
    assert result["chosen"] == 2.0
    assert result["train_ci"]["crosses_zero"] is False
    assert result["cost"]["point"] < 0 and result["cost"]["crosses_zero"] is False
    assert result["pass"] is True


def test_heldout_lot_keeps_S_eq_s_when_the_training_ci_crosses_zero():
    result = oup.heldout_lot(_lot_frame(noisy=True), multiples=[0.0, 2.0])
    assert result["chosen"] == 0.0
    assert "crosses zero" in result["why"]
    assert result["pass"] is False


def test_implied_ordering_cost_inverts_the_eoq_formula():
    rows = []
    for m in (0.0, 2.0):
        rows.append(
            {
                "form": oup.LEGACY[0],
                "granularity": oup.LEGACY[1],
                "lot_multiple": m,
                "target": 0.95,
                "model": "LightGBM",
                "fold": 1,
                "series_id": "A",
                "unit_price": 10.0,
                "reorder_point": 5.0,
                "order_up_to": 5.0 + m * 14.0,
            }
        )
    out = oup.implied_ordering_cost(pd.DataFrame(rows), 0.25, None, 7)
    h = 0.25 / 365
    q, d = 28.0, 14.0 / 7
    assert out.loc[1, "median implied K ($/order)"] == pytest.approx(q**2 * h * 10.0 / (2 * d))
    assert np.isnan(out.loc[0, "median implied K ($/order)"])


def test_scheme_saturation_is_reproducible_and_reports_every_cell_and_target(monkeypatch):
    from reorderpoint import ablate_safety_stock as ab

    monkeypatch.setattr(bt, "MODEL_FACTORIES", {"LightGBM": None})
    panel = _panel()
    forecasts, calib = _forecasts(panel), _calib(panel)
    cells = [("normal", "intermittency"), ("empirical", "intermittency")]
    targets = [0.9, 0.999]
    a = ot.scheme_saturation(_costs(), panel, forecasts, calib, cells, targets)
    b = ot.scheme_saturation(_costs(), panel, forecasts, calib, cells, targets)
    assert len(a) == 4 and set(a["target"]) == set(targets)
    assert a["realised_csl"].between(0, 1).all()
    pd.testing.assert_frame_equal(a, b)  # deterministic
    assert ab is not None


def test_order_quantity_orders_up_to_S_only_while_below_s():
    rp = pd.Series([10.0, 10.0, 10.0])
    on_hand = pd.Series([4.0, 10.0, 12.0])
    S = pd.Series([25.0, 25.0, 25.0])
    assert dec.order_quantity(rp, on_hand).tolist() == [6.0, 0.0, 0.0]  # original S = s
    assert dec.order_quantity(rp, on_hand, S).tolist() == [21.0, 0.0, 0.0]  # S - on_hand, below s


def test_compute_decisions_lot_multiple_matches_the_simulated_rule():
    dates = pd.date_range("2024-01-01", periods=7)
    fc = pd.DataFrame(
        {
            "series_id": "A",
            "date": dates,
            "p10": 1.0,
            "p50": 2.0,
            "p90": 3.0,
        }
    )
    on_hand = pd.Series({"A": 0.0})
    base = dec.compute_decisions(fc, on_hand, 7, 0.95)
    lot = dec.compute_decisions(fc, on_hand, 7, 0.95, lot_multiple=2.0)
    mean = 14.0  # 7 days x p50 = 2
    assert base.loc["A", "order_up_to"] == pytest.approx(base.loc["A", "reorder_point"])
    assert lot.loc["A", "order_up_to"] == pytest.approx(lot.loc["A", "reorder_point"] + 2 * mean)
    assert lot.loc["A", "order_quantity"] == pytest.approx(lot.loc["A", "order_up_to"])
    assert lot.loc["A", "reorder_point"] == pytest.approx(base.loc["A", "reorder_point"])
