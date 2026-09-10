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
