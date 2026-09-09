"""Backtester wiring tests. Real statsforecast models are NOT exercised here (too slow for a
unit test) — evaluate_fold is tested against a monkeypatched dummy model instead, so the merge
and metric-aggregation logic itself is verified without the AutoETS/AutoTheta runtime cost.
"""

import numpy as np
import pandas as pd
import pytest

from reorderpoint import backtest as bt


def _panel(n_days: int, series_ids: list[str], start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for sid in series_ids:
        for i, d in enumerate(dates):
            rows.append({"series_id": sid, "date": d, "y": float(i % 5), "price": 10.0})
    return pd.DataFrame(rows)


def test_make_folds_boundaries():
    panel = _panel(200, ["A"])
    folds = bt.make_folds(panel, horizon=28, n_folds=4)

    assert len(folds) == 4
    assert [f.index for f in folds] == [1, 2, 3, 4]

    last_date = panel["date"].max()
    assert folds[-1].test_end == last_date
    for f in folds:
        assert (f.test_end - f.test_start).days == 27  # 28-day inclusive window
        assert f.train_end == f.test_start - pd.Timedelta(days=1)
    # non-overlapping, chronologically increasing
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test_end < later.test_start


def test_sample_series_respects_n_and_is_deterministic():
    panel = _panel(5, [f"S{i}" for i in range(10)])
    sampled = bt.sample_series(panel, 5, seed=0)
    sampled2 = bt.sample_series(panel, 5, seed=0)
    assert sampled["series_id"].nunique() == 5
    assert set(sampled["series_id"]) == set(sampled2["series_id"])


def test_sample_series_returns_all_when_n_exceeds_available():
    panel = _panel(5, [f"S{i}" for i in range(3)])
    sampled = bt.sample_series(panel, 10)
    assert sampled["series_id"].nunique() == 3
    assert len(sampled) == len(panel)


def test_in_sample_scales_hand_computed():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    train = pd.DataFrame(
        {
            "series_id": ["A"] * 5,
            "date": dates,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    scales = bt._in_sample_scales(train, season_length=1)
    # lag-1 diffs: 1,1,1,1 -> mean abs = 1, mean sq = 1
    assert scales.loc["A", "scale_mae"] == pytest.approx(1.0)
    assert scales.loc["A", "scale_mse"] == pytest.approx(1.0)


def test_in_sample_scales_constant_series_is_nan():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    train = pd.DataFrame({"series_id": ["A"] * 5, "date": dates, "y": [2.0] * 5})
    scales = bt._in_sample_scales(train, season_length=1)
    assert np.isnan(scales.loc["A", "scale_mae"])


def test_fold_revenue_weights():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    train = pd.DataFrame(
        {"series_id": ["A"] * 30, "date": dates, "y": [1.0] * 30, "price": [10.0] * 30}
    )
    weights = bt._fold_revenue_weights(train, horizon=28)
    # last 28 days x (1 unit * $10) = $280
    assert weights.loc["A"] == pytest.approx(280.0)


def test_intermittency():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    train = pd.DataFrame({"series_id": ["A"] * 4, "date": dates, "y": [0.0, 0.0, 1.0, 2.0]})
    rates = bt._intermittency(train)
    assert rates.loc["A"] == pytest.approx(0.5)


class _DummyModel:
    """Predicts a constant P10/P50/P90 for every series/date — enough to check the wiring."""

    def fit(self, train: pd.DataFrame) -> None:
        self._series_ids = train["series_id"].unique()

    def predict_quantiles(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        dates = pd.date_range("2020-06-01", periods=horizon, freq="D")
        rows = [
            {"series_id": sid, "date": d, "p10": 1.0, "p50": 2.0, "p90": 3.0}
            for sid in self._series_ids
            for d in dates
        ]
        return pd.DataFrame(rows)


def test_evaluate_fold_with_dummy_model(monkeypatch):
    monkeypatch.setitem(bt.MODEL_FACTORIES, "Dummy", lambda horizon: _DummyModel())

    train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
    test_dates = pd.date_range("2020-06-01", periods=28, freq="D")
    all_dates = train_dates.union(test_dates)
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * len(all_dates),
            "date": all_dates,
            "y": [2.0] * len(all_dates),  # constant series -> matches dummy p50 exactly
            "price": [10.0] * len(all_dates),
        }
    )
    fold = bt.Fold(
        index=1,
        train_end=train_dates.max(),
        test_start=test_dates.min(),
        test_end=test_dates.max(),
    )
    train = panel[panel["date"] <= fold.train_end]
    # y is constant, so lag-7 diffs are 0 everywhere -> scale is NaN by design (see above test);
    # give it one bump so the scale is defined.
    train = train.copy()
    train.loc[train.index[0], "y"] = 5.0
    scales = bt._in_sample_scales(train, bt.SEASON_LENGTH)
    weights = bt._fold_revenue_weights(train, bt.HORIZON)
    zero_rates = bt._intermittency(train)

    row, detail = bt.evaluate_fold(panel, fold, "Dummy", scales, weights, zero_rates)

    assert row["model"] == "Dummy"
    assert row["n_series"] == 1
    # y=2.0 everywhere in test, dummy predicts p50=2.0 exactly -> zero error
    assert row["mase"] == pytest.approx(0.0)
    assert row["rmsse"] == pytest.approx(0.0)
    # y always within [p10=1, p90=3] -> full coverage
    assert row["coverage_80"] == pytest.approx(1.0)
    assert len(detail) == 1


def test_markdown_table_formats_floats():
    df = pd.DataFrame({"model": ["A"], "mase": [1.23456]})
    table = bt._markdown_table(df, float_cols=("mase",))
    assert "1.235" in table
    assert "| model | mase |" in table
