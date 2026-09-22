"""Hand-computed checks for the newsvendor policy and cost simulation (docs/decision.md)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from reorderpoint import backtest as bt
from reorderpoint import decision as dec
from reorderpoint.config import CostParams

Z80 = norm.ppf(0.9)


def _forecast(series_id: str, p10: list[float], p50: list[float], p90: list[float]) -> pd.DataFrame:
    n = len(p50)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"series_id": [series_id] * n, "date": dates, "p10": p10, "p50": p50, "p90": p90}
    )


def test_lead_time_demand_stats_hand_computed():
    # 3 identical days: p50=10, p10=8, p90=12 -> daily_std = (12-8)/(2*Z80) = 2/Z80
    fc = _forecast("A", p10=[8, 8, 8], p50=[10, 10, 10], p90=[12, 12, 12])
    stats = dec.lead_time_demand_stats(fc, lead_time_days=3)
    assert stats.loc["A", "mean"] == pytest.approx(30.0)
    daily_std = 2 / Z80
    assert stats.loc["A", "std"] == pytest.approx(np.sqrt(3) * daily_std)


def test_lead_time_demand_stats_only_uses_first_n_days():
    fc = _forecast("A", p10=[8, 0, 0], p50=[10, 100, 100], p90=[12, 200, 200])
    stats = dec.lead_time_demand_stats(fc, lead_time_days=1)
    assert stats.loc["A", "mean"] == pytest.approx(10.0)


def test_reorder_decision_hand_computed():
    stats = pd.DataFrame({"mean": [30.0], "std": [2.0]}, index=pd.Index(["A"], name="series_id"))
    out = dec.reorder_decision(stats, service_level=0.95)
    z = norm.ppf(0.95)
    assert out.loc["A", "safety_stock"] == pytest.approx(z * 2.0)
    assert out.loc["A", "reorder_point"] == pytest.approx(30.0 + z * 2.0)


def test_order_quantity_below_reorder_point():
    s = pd.Series([50.0, 50.0], index=["A", "B"])
    on_hand = pd.Series([20.0, 60.0], index=["A", "B"])
    qty = dec.order_quantity(s, on_hand)
    assert qty["A"] == pytest.approx(30.0)
    assert qty["B"] == pytest.approx(0.0)


def test_stockout_probability_hand_computed():
    on_hand = pd.Series([30.0], index=["A"])
    mean = pd.Series([30.0], index=["A"])
    std = pd.Series([2.0], index=["A"])
    prob = dec.stockout_probability(on_hand, mean, std)
    assert prob["A"] == pytest.approx(0.5)  # on-hand exactly at mean


def test_stockout_probability_zero_std_deterministic():
    on_hand = pd.Series([10.0, 5.0], index=["A", "B"])
    mean = pd.Series([8.0, 8.0], index=["A", "B"])
    std = pd.Series([0.0, 0.0], index=["A", "B"])
    prob = dec.stockout_probability(on_hand, mean, std)
    assert prob["A"] == pytest.approx(0.0)
    assert prob["B"] == pytest.approx(1.0)


def test_compute_decisions_end_to_end():
    fc = _forecast("A", p10=[8, 8, 8], p50=[10, 10, 10], p90=[12, 12, 12])
    on_hand = pd.Series([5.0], index=["A"])
    out = dec.compute_decisions(fc, on_hand, lead_time_days=3, service_level=0.95)
    assert out.loc["A", "on_hand"] == 5.0
    assert out.loc["A", "reorder_point"] > out.loc["A", "mean"]
    assert out.loc["A", "order_quantity"] == pytest.approx(out.loc["A", "reorder_point"] - 5.0)
    assert isinstance(out.loc["A", "rationale"], str)
    assert "order" in out.loc["A", "rationale"].lower()


def test_unit_cost_from_price_uses_training_mean():
    train = pd.DataFrame(
        {
            "series_id": ["A", "A", "B"],
            "price": [10.0, 12.0, np.nan],
        }
    )
    out = dec.unit_cost_from_price(train)
    assert out["A"] == pytest.approx(11.0)
    assert out["B"] == pytest.approx(11.0)  # missing price falls back to global mean


def _costs(**overrides) -> CostParams:
    base = dict(
        holding_cost_rate=0.02,
        stockout_penalty_per_unit=5.0,
        lead_time_days=2,
        service_level_target=0.95,
    )
    base.update(overrides)
    return CostParams(**base)


def test_simulate_series_no_stockout_no_reorder():
    # on_hand comfortably above demand every day, s low enough it's never triggered
    demand = np.array([1.0, 1.0, 1.0, 1.0])
    result = dec.simulate_series(
        demand, s=0.0, S=0.0, on_hand_start=10.0, lead_time_days=2, unit_cost=2.0, costs=_costs()
    )
    assert result.units_shipped == pytest.approx(4.0)
    assert result.stockout_cost == pytest.approx(0.0)
    assert result.fill_rate == pytest.approx(1.0)
    # on_hand after each day: 9,8,7,6 -> holding = 0.02*2.0*(9+8+7+6) = 1.2
    assert result.holding_cost == pytest.approx(0.02 * 2.0 * 30)


def test_simulate_series_lost_sale_when_stock_runs_out():
    demand = np.array([10.0, 10.0])
    result = dec.simulate_series(
        demand, s=0.0, S=0.0, on_hand_start=5.0, lead_time_days=2, unit_cost=1.0, costs=_costs()
    )
    # day1: ship 5 (all on-hand), lost 5; day2: on_hand=0, ship 0, lost 10
    assert result.units_shipped == pytest.approx(5.0)
    assert result.units_demanded == pytest.approx(20.0)
    assert result.stockout_cost == pytest.approx(15.0 * 5.0)  # 15 lost units * penalty
    assert result.fill_rate == pytest.approx(5.0 / 20.0)


def test_simulate_series_reorder_arrives_after_lead_time():
    # s=S=8: starts at on_hand_start=8 (>= s, no immediate order). Day1 demand=5 -> on_hand=3 < s
    # -> order (S-3)=5 placed, arrives after lead_time_days=2 i.e. on day index 1+2=3.
    demand = np.array([5.0, 0.0, 0.0, 0.0])
    result = dec.simulate_series(
        demand, s=8.0, S=8.0, on_hand_start=8.0, lead_time_days=2, unit_cost=1.0, costs=_costs()
    )
    # day0: ship5,on_hand=3(<8 -> order 5, arrives day2)
    # day1: on_hand stays 3
    # day2: order arrives -> on_hand=8
    # day3: on_hand=8
    assert result.stockout_cost == pytest.approx(0.0)
    assert result.units_shipped == pytest.approx(5.0)


def test_lead_time_forecast_residuals_hand_computed():
    fc = _forecast("A", p10=[1, 1], p50=[10, 10], p90=[20, 20])
    actual = pd.DataFrame({"series_id": ["A", "A"], "date": fc["date"], "y": [12.0, 8.0]})
    residuals = dec.lead_time_forecast_residuals(fc, actual, lead_time_days=2)
    # predicted sum = 20, actual sum = 20 -> residual 0
    assert residuals["A"] == pytest.approx(0.0)


def test_lead_time_forecast_residuals_only_uses_first_n_days():
    fc = _forecast("A", p10=[1, 1, 1], p50=[10, 10, 999], p90=[20, 20, 20])
    actual = pd.DataFrame({"series_id": ["A"] * 3, "date": fc["date"], "y": [10.0, 10.0, 999.0]})
    residuals = dec.lead_time_forecast_residuals(fc, actual, lead_time_days=2)
    # only first 2 days used: predicted=20, actual=20 -> 0, ignoring day 3's huge values
    assert residuals["A"] == pytest.approx(0.0)


def test_empirical_lead_time_std_pools_by_intermittency_bucket():
    # 4 "regular" series with residuals [1,-1,1,-1] -> std=sqrt of variance of that set
    # 4 "intermittent" series with residuals [10,-10,10,-10] -> much larger std
    series_ids = [f"R{i}" for i in range(4)] + [f"I{i}" for i in range(4)]
    residuals = pd.Series([1.0, -1.0, 1.0, -1.0, 10.0, -10.0, 10.0, -10.0], index=series_ids)
    zero_rates = pd.Series([0.1] * 4 + [0.9] * 4, index=series_ids)

    out = dec.empirical_lead_time_std(residuals, zero_rates, threshold=0.5)

    regular_std = residuals.loc[["R0", "R1", "R2", "R3"]].std(ddof=1)
    intermittent_std = residuals.loc[["I0", "I1", "I2", "I3"]].std(ddof=1)
    assert out["R0"] == pytest.approx(regular_std)
    assert out["I0"] == pytest.approx(intermittent_std)
    assert out["I0"] > out["R0"]


class _DummyModel:
    """Constant P10=1/P50=2/P90=3 forecast for every series/date — same shape as
    test_backtest.py's dummy, reused here to check the cost-simulation wiring."""

    def fit(self, train: pd.DataFrame) -> None:
        self._series_ids = train["series_id"].unique()

    def predict_quantiles(self, horizon: int, future_exog=None) -> pd.DataFrame:
        dates = pd.date_range("2020-06-01", periods=horizon, freq="D")
        rows = [
            {"series_id": sid, "date": d, "p10": 1.0, "p50": 2.0, "p90": 3.0}
            for sid in self._series_ids
            for d in dates
        ]
        return pd.DataFrame(rows)


def test_evaluate_fold_cost_with_dummy_model(monkeypatch):
    monkeypatch.setitem(bt.MODEL_FACTORIES, "Dummy", lambda horizon: _DummyModel())

    train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # test window == lead_time_days: on_hand_start = reorder_point = lead_time_mean +
    # safety_stock, and actual demand matches the dummy's p50 exactly, so cumulative demand
    # over the window can never exceed on_hand_start — zero stockout is guaranteed by
    # construction here (not true for windows longer than lead_time_days; see
    # test_simulate_series_lost_sale_when_stock_runs_out for that case).
    test_dates = pd.date_range("2020-06-01", periods=2, freq="D")
    all_dates = train_dates.union(test_dates)
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(all_dates),
            "date": all_dates,
            "y": [2.0] * len(all_dates),  # constant, matches dummy p50 exactly
            "price": [10.0] * len(all_dates),
        }
    )
    fold = bt.Fold(
        index=1, train_end=train_dates.max(), test_start=test_dates.min(), test_end=test_dates.max()
    )
    costs = _costs(lead_time_days=2)

    row, detail, next_std = dec.evaluate_fold_cost(panel, fold, "Dummy", costs)

    assert row["model"] == "Dummy"
    assert row["n_series"] == 1
    assert row["stockout_cost"] == pytest.approx(0.0)
    assert row["fill_rate"] == pytest.approx(1.0)
    assert row["total_cost"] == pytest.approx(row["holding_cost"])
    assert len(detail) == 1
    assert "A" in next_std.index


class _NaNForOneSeriesModel:
    """Like _DummyModel, but one series' forecast is NaN — simulates a model that fails to
    produce a valid prediction for a specific series (e.g. insufficient history)."""

    def fit(self, train: pd.DataFrame) -> None:
        self._series_ids = train["series_id"].unique()

    def predict_quantiles(self, horizon: int, future_exog=None) -> pd.DataFrame:
        dates = pd.date_range("2020-06-01", periods=horizon, freq="D")
        rows = []
        for sid in self._series_ids:
            p10, p50, p90 = (float("nan"),) * 3 if sid == "BAD" else (1.0, 2.0, 3.0)
            rows.extend(
                {"series_id": sid, "date": d, "p10": p10, "p50": p50, "p90": p90} for d in dates
            )
        return pd.DataFrame(rows)


def test_evaluate_fold_cost_skips_series_with_nan_reorder_point(monkeypatch):
    monkeypatch.setitem(bt.MODEL_FACTORIES, "PartialNaN", lambda horizon: _NaNForOneSeriesModel())

    train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
    test_dates = pd.date_range("2020-06-01", periods=2, freq="D")
    all_dates = train_dates.union(test_dates)
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "series_id": [sid] * len(all_dates),
                    "date": all_dates,
                    "y": [2.0] * len(all_dates),
                    "price": [10.0] * len(all_dates),
                }
            )
            for sid in ("GOOD", "BAD")
        ],
        ignore_index=True,
    )
    fold = bt.Fold(
        index=1, train_end=train_dates.max(), test_start=test_dates.min(), test_end=test_dates.max()
    )
    costs = _costs(lead_time_days=2)

    row, detail, _next_std = dec.evaluate_fold_cost(panel, fold, "PartialNaN", costs)

    # BAD's NaN reorder_point must not silently "never reorder" through the simulation —
    # it's excluded entirely, not counted as a (spuriously perfect-looking) zero-stockout series.
    assert row["n_series"] == 1
    assert list(detail["series_id"]) == ["GOOD"]


def test_evaluate_fold_cost_uses_lead_time_std_override(monkeypatch):
    # Dummy's own P10/P90 spread implies a large std; passing a near-zero override should shrink
    # the resulting reorder_point (and thus on_hand_start) close to the mean, not the quantile-std.
    monkeypatch.setitem(bt.MODEL_FACTORIES, "Dummy", lambda horizon: _DummyModel())

    train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
    test_dates = pd.date_range("2020-06-01", periods=2, freq="D")
    all_dates = train_dates.union(test_dates)
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(all_dates),
            "date": all_dates,
            "y": [2.0] * len(all_dates),
            "price": [10.0] * len(all_dates),
        }
    )
    fold = bt.Fold(
        index=1, train_end=train_dates.max(), test_start=test_dates.min(), test_end=test_dates.max()
    )
    costs = _costs(lead_time_days=2)

    row_default, _, _ = dec.evaluate_fold_cost(panel, fold, "Dummy", costs)
    tiny_std_override = pd.Series([1e-9], index=["A"])
    row_override, _, _ = dec.evaluate_fold_cost(
        panel, fold, "Dummy", costs, lead_time_std_override=tiny_std_override
    )

    # smaller safety stock -> smaller on_hand_start -> holding cost drops
    assert row_override["holding_cost"] < row_default["holding_cost"]


def test_evaluate_fold_cost_returns_next_fold_std_from_own_residuals(monkeypatch):
    class _BiasedModel:
        """Predicts p50=0 always; actual y=2/day -> residual (actual-predicted) is always +2
        per lead-time day, so a 2-day lead time gives a residual of +4 every time -> std=0
        (constant residual) once enough series/folds exist. Used to check the wiring returns
        *something* derived from this fold's real (forecast, actual) pair, not the quantile std.
        """

        def fit(self, train):
            self._series_ids = train["series_id"].unique()

        def predict_quantiles(self, horizon, future_exog=None):
            dates = pd.date_range("2020-06-01", periods=horizon, freq="D")
            rows = [
                {"series_id": sid, "date": d, "p10": 0.0, "p50": 0.0, "p90": 0.0}
                for sid in self._series_ids
                for d in dates
            ]
            return pd.DataFrame(rows)

    train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
    test_dates = pd.date_range("2020-06-01", periods=2, freq="D")
    all_dates = train_dates.union(test_dates)
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(all_dates),
            "date": all_dates,
            "y": [2.0] * len(all_dates),
            "price": [10.0] * len(all_dates),
        }
    )
    fold = bt.Fold(
        index=1, train_end=train_dates.max(), test_start=test_dates.min(), test_end=test_dates.max()
    )
    costs = _costs(lead_time_days=2)

    monkeypatch.setitem(bt.MODEL_FACTORIES, "Biased", lambda horizon: _BiasedModel())
    _, _, next_std = dec.evaluate_fold_cost(panel, fold, "Biased", costs)

    # p50=0 predicted, actual=2/day -> residual = +4 for the single series in this fold; pooled
    # std of a single-series bucket with one observation is NaN (ddof=1 needs >=2 points) and
    # empirical_lead_time_std falls back to the overall std in that case (also NaN with n=1) —
    # so it should be finite-or-NaN, not raise, and the series must still be present.
    assert "A" in next_std.index


def test_run_decision_backtest_and_report_smoke(monkeypatch):
    monkeypatch.setattr(bt, "MODEL_FACTORIES", {"Dummy": lambda horizon: _DummyModel()})
    monkeypatch.setattr(bt, "N_SERIES_SAMPLE", 1)
    monkeypatch.setattr(bt, "HORIZON", 4)
    monkeypatch.setattr(bt, "N_FOLDS", 1)

    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(dates),
            "date": dates,
            "y": [2.0] * len(dates),
            "price": [10.0] * len(dates),
        }
    )
    costs = _costs(lead_time_days=2)

    results, detail = dec.run_decision_backtest(panel, costs)
    assert set(results["model"]) == {"Dummy"}
    assert "total_cost" in results.columns

    report = dec.write_decision_report(results, costs)
    assert "Decision" in report or "decision" in report
    assert "total_cost" in report or "Total cost" in report.lower()


def test_run_decision_backtest_lot_multiples_covers_every_lot_from_one_fit(monkeypatch):
    """`run_deep_backtest.py` queues `S = s` and `S = s + 2 x lead-time demand` from a single
    `run_decision_backtest` call (`lot_multiples=(0.0, 2.0)`) so an expensive model (NBEATS) is
    fit only once — this pins that both lots actually reach `_simulate_policy`, tagged correctly
    in both `results` and `detail`, for every fold/model/policy row."""
    monkeypatch.setattr(bt, "MODEL_FACTORIES", {"Dummy": lambda horizon: _DummyModel()})
    monkeypatch.setattr(bt, "N_SERIES_SAMPLE", 1)
    monkeypatch.setattr(bt, "HORIZON", 4)
    monkeypatch.setattr(bt, "N_FOLDS", 2)

    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(dates),
            "date": dates,
            "y": [2.0] * len(dates),
            "price": [10.0] * len(dates),
        }
    )
    costs = _costs(lead_time_days=2)

    results, detail = dec.run_decision_backtest(panel, costs, lot_multiples=(0.0, 2.0))

    assert set(results["lot_multiple"]) == {0.0, 2.0}
    assert set(detail["lot_multiple"]) == {0.0, 2.0}
    base_detail = detail[detail["lot_multiple"] == 0.0]
    lot_detail = detail[detail["lot_multiple"] == 2.0]
    assert len(lot_detail) == len(base_detail) > 1  # every fold/model/policy row got both lots
    # a positive mean forecast makes S strictly above s under the lot policy, everywhere
    assert (lot_detail["order_up_to"] > lot_detail["reorder_point"]).all()
    assert (base_detail["order_up_to"] == base_detail["reorder_point"]).all()


# --- v1.1 Task 2: cycle service level and the fill-rate-targeted policy ---


def test_simulate_series_counts_a_clean_cycle():
    # s=S=8, start at 8. day0: ship 5 -> on_hand 3 < 8, order placed, arrives day 2.
    # No demand while the order is in transit, so the cycle closes without a stockout.
    demand = np.array([5.0, 0.0, 0.0, 0.0])
    result = dec.simulate_series(
        demand, s=8.0, S=8.0, on_hand_start=8.0, lead_time_days=2, unit_cost=1.0, costs=_costs()
    )
    assert result.cycles == 1
    assert result.stockout_cycles == 0


def test_simulate_series_counts_a_stockout_cycle():
    # Order placed at the end of day 0 (on_hand 0 < s), arrives day 2. Day 1's demand of 5 is
    # entirely lost while that order is in transit -> the cycle is a stockout cycle.
    demand = np.array([8.0, 5.0, 0.0, 0.0])
    result = dec.simulate_series(
        demand, s=8.0, S=8.0, on_hand_start=8.0, lead_time_days=2, unit_cost=1.0, costs=_costs()
    )
    assert result.cycles == 1
    assert result.stockout_cycles == 1
    assert result.fill_rate == pytest.approx(8.0 / 13.0)


def test_simulate_series_open_cycle_at_horizon_end_is_not_counted():
    # Order placed day 0, lead time 5, but the horizon ends at day 3 — it never lands. An
    # unfinished cycle is not evidence either way and must not be scored as a success.
    demand = np.array([8.0, 0.0, 0.0, 0.0])
    result = dec.simulate_series(
        demand, s=8.0, S=8.0, on_hand_start=8.0, lead_time_days=5, unit_cost=1.0, costs=_costs()
    )
    assert result.cycles == 0
    assert result.stockout_cycles == 0


def test_simulate_series_lost_sales_outside_any_cycle_do_not_touch_csl():
    # s=S=0 never triggers an order, so there are no cycles at all, yet demand is lost.
    # Fill rate must register that; CSL has no cycle to register it in.
    demand = np.array([4.0, 4.0])
    result = dec.simulate_series(
        demand, s=0.0, S=0.0, on_hand_start=0.0, lead_time_days=2, unit_cost=1.0, costs=_costs()
    )
    assert result.cycles == 0
    assert result.fill_rate == pytest.approx(0.0)


def test_standard_loss_matches_known_values():
    # G(0) = phi(0) = 1/sqrt(2*pi); G is strictly decreasing and positive.
    assert dec.standard_loss(0.0) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert dec.standard_loss(1.0) == pytest.approx(norm.pdf(1.0) - 1.0 * (1 - norm.cdf(1.0)))
    assert dec.standard_loss(3.0) > 0
    assert dec.standard_loss(3.0) < dec.standard_loss(1.0) < dec.standard_loss(-1.0)


def test_fill_rate_k_inverts_the_loss_relationship():
    # Round-trip: pick k, derive the fill rate it implies, and check the solver recovers k.
    q, sigma = 20.0, 8.0
    for k_true in [-0.5, 0.0, 1.0, 2.0]:
        fill = 1 - sigma * dec.standard_loss(k_true) / q
        assert dec.fill_rate_k(float(fill), q, sigma) == pytest.approx(k_true, abs=1e-6)


def test_fill_rate_k_needs_more_safety_stock_for_a_smaller_order_quantity():
    # The Task 2 mechanism: with S = s, Q is small, and a small Q forces a larger k for the
    # same fill rate.
    big_q = dec.fill_rate_k(0.95, order_quantity=100.0, lead_time_std=10.0)
    small_q = dec.fill_rate_k(0.95, order_quantity=10.0, lead_time_std=10.0)
    assert small_q > big_q


def test_fill_rate_k_degenerate_inputs_return_zero():
    assert dec.fill_rate_k(0.95, order_quantity=0.0, lead_time_std=5.0) == 0.0
    assert dec.fill_rate_k(0.95, order_quantity=5.0, lead_time_std=0.0) == 0.0
    assert dec.fill_rate_k(0.95, order_quantity=5.0, lead_time_std=np.nan) == 0.0


def test_fill_rate_k_rejects_out_of_range_target():
    with pytest.raises(ValueError):
        dec.fill_rate_k(1.0, order_quantity=5.0, lead_time_std=1.0)


def test_reorder_decision_fill_rate_sizes_from_the_loss_function():
    stats = pd.DataFrame({"mean": [20.0], "std": [8.0]}, index=pd.Index(["A"], name="series_id"))
    out = dec.reorder_decision_fill_rate(stats, target_fill_rate=0.95)
    k = dec.fill_rate_k(0.95, order_quantity=20.0, lead_time_std=8.0)
    assert out.loc["A", "safety_stock"] == pytest.approx(k * 8.0)
    assert out.loc["A", "reorder_point"] == pytest.approx(20.0 + k * 8.0)


def test_fill_rate_policy_vs_csl_policy_depends_on_the_sigma_to_q_ratio():
    """Which policy is more conservative is not fixed — it turns on σ_LT/Q.

    The two targets answer different questions, so at the same nominal 95% neither dominates:
    G(k) = (1 − FR)·Q/σ means a series whose lead-time spread is small next to its per-cycle
    demand needs *less* than z = 1.645 to meet a 95% fill rate, while a spiky series whose σ
    dwarfs Q needs considerably more. Both regimes exist in this panel, which is why the pooled
    fill-rate and CSL numbers can't be read off one another.
    """
    stats = pd.DataFrame(
        {"mean": [10.0, 10.0], "std": [9.0, 60.0]},
        index=pd.Index(["moderate", "spiky"], name="series_id"),
    )
    csl = dec.reorder_decision(stats, service_level=0.95)
    fr = dec.reorder_decision_fill_rate(stats, target_fill_rate=0.95)

    assert fr.loc["moderate", "reorder_point"] < csl.loc["moderate", "reorder_point"]
    assert fr.loc["spiky", "reorder_point"] > csl.loc["spiky", "reorder_point"]


def test_simulate_series_records_realised_order_quantities():
    # s=S=8, start 8. day0 demand 5 -> on_hand 3, order 5 placed (arrives day 2).
    # day2 order lands -> on_hand 8; day3 demand 2 -> on_hand 6 < 8, order 2 placed.
    demand = np.array([5.0, 0.0, 0.0, 2.0])
    result = dec.simulate_series(
        demand, s=8.0, S=8.0, on_hand_start=8.0, lead_time_days=2, unit_cost=1.0, costs=_costs()
    )
    assert result.orders == 2
    assert result.order_qty_total == pytest.approx(7.0)  # 5 + 2
    assert result.order_qty_sq_total == pytest.approx(29.0)  # 25 + 4
    assert result.orders_below_one_unit == 0


def test_simulate_series_flags_sub_unit_orders():
    # A reorder point just above on-hand produces a fractional order — the regime that makes Q a
    # poor stand-in for a lot size on intermittent demand.
    demand = np.array([0.5, 0.0, 0.0])
    result = dec.simulate_series(
        demand, s=1.0, S=1.0, on_hand_start=1.0, lead_time_days=2, unit_cost=1.0, costs=_costs()
    )
    assert result.orders == 1
    assert result.orders_below_one_unit == 1


def test_order_quantity_stats_pooled_over_orders():
    detail = pd.DataFrame(
        {
            # series A: 2 orders of total 10 (sq 50 -> 5 and 5); series B: 1 order of 20
            "orders": [2, 1],
            "order_qty_total": [10.0, 20.0],
            "order_qty_sq_total": [50.0, 400.0],
            "orders_below_one_unit": [1, 0],
            "lead_time_std": [10.0, 10.0],
        }
    )
    stats = dec._order_quantity_stats(detail)
    assert stats["q_mean"] == pytest.approx(10.0)  # 30 units / 3 orders
    assert stats["q_frac_below_one"] == pytest.approx(1 / 3)
    # per-series mean Q is 5 (A) and 20 (B) -> sigma/Q of 2.0 and 0.5 -> median 1.25
    assert stats["sigma_over_q"] == pytest.approx(1.25)


def test_order_quantity_stats_with_no_orders_is_nan_not_zero():
    detail = pd.DataFrame(
        {
            "orders": [0],
            "order_qty_total": [0.0],
            "order_qty_sq_total": [0.0],
            "orders_below_one_unit": [0],
            "lead_time_std": [5.0],
        }
    )
    stats = dec._order_quantity_stats(detail)
    assert np.isnan(stats["q_mean"])
    assert np.isnan(stats["q_cv"])
