"""Tests for the hierarchical reconciliation experiment (Phase 5).

build_hierarchy/reconcile_forecasts wrap hierarchicalforecast directly on tiny synthetic data —
those calls are fast and deterministic, so they're exercised for real (no mocking), matching this
project's own convention (see metrics.py, backtest.py's non-model helpers). Only the real
statsforecast model fit is skipped in tests, same reasoning as test_backtest.py.
"""

import pandas as pd
import pytest

from reorderpoint import backtest as bt
from reorderpoint import reconcile as rec


def _bottom_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rows = []
    for state, store, cat, dept, item, base in [
        ("CA", "CA_1", "FOODS", "FOODS_1", "F1", 1.0),
        ("CA", "CA_1", "HOBBIES", "HOBBIES_1", "H1", 3.0),
        ("TX", "TX_1", "FOODS", "FOODS_1", "F2", 5.0),
        ("TX", "TX_1", "HOBBIES", "HOBBIES_1", "H2", 7.0),
    ]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "state_id": state,
                    "store_id": store,
                    "cat_id": cat,
                    "dept_id": dept,
                    "item_id": item,
                    "date": d,
                    "y": base + i,
                }
            )
    return pd.DataFrame(rows)


def test_build_hierarchy_produces_every_level():
    panel = _bottom_panel()
    Y_df, S_df, tags = rec.build_hierarchy(panel)

    assert {"series_id", "date", "y"}.issubset(Y_df.columns)
    assert set(tags.keys()) == set(rec.LEVEL_NAMES)
    # bottom level has the 4 original series; top level ("state_id") has 2 (CA, TX)
    assert len(tags[rec.LEVEL_NAMES[-1]]) == 4
    assert len(tags[rec.LEVEL_NAMES[0]]) == 2


def test_build_hierarchy_aggregates_values_correctly():
    panel = _bottom_panel()
    Y_df, S_df, tags = rec.build_hierarchy(panel)
    # CA total on day 1 = F1(1.0) + H1(3.0) = 4.0
    ca_row = Y_df[(Y_df["series_id"] == "CA") & (Y_df["date"] == pd.Timestamp("2020-01-01"))]
    assert ca_row["y"].iloc[0] == pytest.approx(4.0)


def test_level_lookup_maps_node_to_its_level():
    panel = _bottom_panel()
    _, _, tags = rec.build_hierarchy(panel)
    lookup = rec.level_lookup(tags)
    assert lookup["CA"] == "state_id"
    assert lookup["CA/CA_1/FOODS/FOODS_1/F1"] == rec.LEVEL_NAMES[-1]


def test_reconcile_forecasts_adds_bottomup_and_mintrace_columns():
    panel = _bottom_panel()
    Y_df, S_df, tags = rec.build_hierarchy(panel)

    # fabricated "base forecast": actual + 1, for every node/date — stands in for a real model's
    # point forecast without paying for a real statsforecast fit (too slow for a unit test, same
    # reasoning as test_backtest.py's dummy model).
    forecast_df = Y_df.rename(columns={"y": rec.MODEL_NAME}).copy()
    forecast_df[rec.MODEL_NAME] = forecast_df[rec.MODEL_NAME] + 1
    fitted_df = Y_df.copy()
    fitted_df[rec.MODEL_NAME] = fitted_df["y"] + 0.5

    reconciled = rec.reconcile_forecasts(forecast_df, fitted_df, S_df, tags)

    assert f"{rec.MODEL_NAME}/BottomUp" in reconciled.columns
    assert any(c.startswith(f"{rec.MODEL_NAME}/MinTrace") for c in reconciled.columns)
    # bottom-up on a bottom-level node is a no-op (nothing to sum up from below it)
    bottom_id = "CA/CA_1/FOODS/FOODS_1/F1"
    row = reconciled[reconciled["series_id"] == bottom_id].iloc[0]
    assert row[f"{rec.MODEL_NAME}/BottomUp"] == pytest.approx(row[rec.MODEL_NAME])


def _wide_bottom_panel(n_days: int) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for state, store, cat, dept, item in [
        ("CA", "CA_1", "FOODS", "FOODS_1", "F1"),
        ("CA", "CA_1", "HOBBIES", "HOBBIES_1", "H1"),
        ("TX", "TX_1", "FOODS", "FOODS_1", "F2"),
        ("TX", "TX_1", "HOBBIES", "HOBBIES_1", "H2"),
    ]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "series_id": f"{item}_{store}",
                    "state_id": state,
                    "store_id": store,
                    "cat_id": cat,
                    "dept_id": dept,
                    "item_id": item,
                    "date": d,
                    "y": float((i % 5) + 1),
                }
            )
    return pd.DataFrame(rows)


def _fake_forecast_fn(train: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stands in for `fit_forecast_with_fitted`: predicts each node's mean training y for every
    future day, and reuses (a slightly perturbed) actual y as the "fitted" in-sample value —
    avoids a real (slow) statsforecast fit in this wiring test, same reasoning as
    test_backtest.py's dummy model.
    """
    nodes = train["series_id"].unique()
    means = train.groupby("series_id")["y"].mean()
    future_dates = pd.date_range(
        train["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    forecast_df = pd.DataFrame(
        [{"series_id": n, "date": d, rec.MODEL_NAME: means[n]} for n in nodes for d in future_dates]
    )
    fitted_df = train[["series_id", "date", "y"]].copy()
    fitted_df[rec.MODEL_NAME] = fitted_df["y"] * 0.9
    return forecast_df, fitted_df


def test_evaluate_fold_reconciliation_wiring():
    panel = _wide_bottom_panel(60)
    Y_df, S_df, tags = rec.build_hierarchy(panel)
    fold = bt.Fold(
        index=1,
        train_end=pd.Timestamp("2020-02-15"),
        test_start=pd.Timestamp("2020-02-16"),
        test_end=pd.Timestamp("2020-02-22"),
    )

    result = rec.evaluate_fold_reconciliation(Y_df, fold, S_df, tags, forecast_fn=_fake_forecast_fn)

    assert set(result["method"]) == {"base", "BottomUp", "MinTrace"}
    assert set(result["level"]) == set(rec.LEVEL_NAMES)
    assert (result["fold"] == 1).all()


def test_run_reconciliation_experiment_and_report_smoke(monkeypatch):
    monkeypatch.setattr(rec, "N_SERIES_SAMPLE", 4)
    monkeypatch.setattr(rec, "STORE_IDS", ["CA_1", "TX_1"])
    monkeypatch.setattr(rec, "HORIZON", 7)
    monkeypatch.setattr(rec, "N_FOLDS", 1)

    panel = _wide_bottom_panel(60)

    results = rec.run_reconciliation_experiment(panel, forecast_fn=_fake_forecast_fn)
    assert {"level", "method", "mase", "rmsse", "n_nodes", "fold"}.issubset(results.columns)

    report = rec.write_reconciliation_report(results)
    assert "reconciliation" in report.lower()
    assert "does reconciliation help" in report.lower()


def test_write_reconciliation_report_explains_bottom_level_noise():
    rows = []
    for level in rec.LEVEL_NAMES:
        for method, mase_val in [("base", 1.0), ("BottomUp", 1.0), ("MinTrace", 1.0)]:
            rows.append(
                {"level": level, "method": method, "mase": mase_val, "rmsse": 1.0, "n_nodes": 5}
            )
    # override: bottom level's base MASE is 4x the top level's -> noisier bottom-level forecasts
    for row in rows:
        if row["level"] == rec.LEVEL_NAMES[-1] and row["method"] == "base":
            row["mase"] = 4.0
        if row["level"] == rec.LEVEL_NAMES[0] and row["method"] == "base":
            row["mase"] = 1.0
    results = pd.DataFrame(rows)

    report = rec.write_reconciliation_report(results)

    assert "4.0" in report or "4.00" in report
    assert "noisier" in report.lower() or "noise" in report.lower()


def test_metrics_by_level_hand_computed():
    # 2 nodes, one at each of 2 levels, with known errors against known in-sample scales
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    train = pd.DataFrame(
        {
            "series_id": ["state_node"] * 5 + ["bottom_node"] * 5,
            "date": list(dates) * 2,
            "y": [10.0, 11.0, 12.0, 13.0, 14.0] + [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    test_dates = pd.date_range("2020-01-06", periods=2, freq="D")
    merged = pd.DataFrame(
        {
            "series_id": ["state_node", "state_node", "bottom_node", "bottom_node"],
            "date": list(test_dates) * 2,
            "y": [15.0, 16.0, 1.0, 1.0],
            "forecast": [16.0, 16.0, 1.0, 2.0],  # state_node off by 1 & 0; bottom off by 0 & 1
        }
    )
    lookup = pd.Series({"state_node": "state_id", "bottom_node": "state_id/store_id"})

    out = rec.metrics_by_level(merged, train, lookup, forecast_col="forecast", season_length=1)

    assert set(out["level"]) == {"state_id", "state_id/store_id"}
    state_row = out[out["level"] == "state_id"].iloc[0]
    bottom_row = out[out["level"] == "state_id/store_id"].iloc[0]
    # state_node: train lag-1 diffs are all 1 -> scale=1; abs errors [1,0] -> mase=0.5
    assert state_row["mase"] == pytest.approx(0.5)
    # bottom_node: train is constant -> in-sample scale is 0 -> NaN -> mase is NaN
    assert pd.isna(bottom_row["mase"])
    assert bottom_row["n_nodes"] == 1
