"""Divergence metric tests on hand-built forecast frames — no models are fit here."""

import numpy as np
import pandas as pd
import pytest

from reorderpoint import divergence as dv


def _wide(values: dict[str, list[float]], y: list[float], n_series: int = 1) -> pd.DataFrame:
    n_days = len(y) // n_series
    idx = pd.MultiIndex.from_tuples(
        [
            (1, f"S{s}", pd.Timestamp("2020-01-01") + pd.Timedelta(days=d))
            for s in range(n_series)
            for d in range(n_days)
        ],
        names=["fold", "series_id", "date"],
    )
    df = pd.DataFrame(values, index=idx)
    df["y"] = y
    df["zero_rate"] = 0.9
    return df


def test_relative_total_difference_treats_two_zero_forecasts_as_identical():
    a = pd.Series([0.0, 100.0, 10.0])
    b = pd.Series([0.0, 104.0, 20.0])
    out = dv.relative_total_difference(a, b)
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == pytest.approx(4 / 102)
    assert out.iloc[2] == pytest.approx(10 / 15)


def test_identical_models_are_perfectly_correlated_and_always_close():
    vals = [1.0, 2.0, 3.0, 0.0, 5.0, 6.0]
    wide = _wide({"A": vals, "B": vals}, y=[1.0] * 6, n_series=2)

    corr = dv.pairwise_correlations(wide, ["A", "B"]).iloc[0]
    assert corr["pearson_pooled"] == pytest.approx(1.0)
    assert corr["pearson_within_series"] == pytest.approx(1.0)

    mad = dv.mean_abs_differences(wide, ["A", "B"]).iloc[0]
    assert mad["mean_abs_diff"] == 0.0

    _, summary = dv.close_fractions(wide, ["A", "B"])
    assert summary["all_pairs_close"] == 1.0


def test_pooled_correlation_can_be_high_while_within_series_shape_disagrees():
    # Two series at very different levels; within each, A and B move in opposite directions.
    # Pooled correlation is driven by the level gap — the within-series column must expose it.
    a = [1.0, 2.0, 1.0, 2.0, 101.0, 102.0, 101.0, 102.0]
    b = [2.0, 1.0, 2.0, 1.0, 102.0, 101.0, 102.0, 101.0]
    wide = _wide({"A": a, "B": b}, y=[1.0] * 8, n_series=2)

    corr = dv.pairwise_correlations(wide, ["A", "B"]).iloc[0]
    assert corr["pearson_pooled"] > 0.99
    assert corr["pearson_within_series"] == pytest.approx(-1.0)


def test_flat_forecast_has_undefined_within_series_correlation():
    wide = _wide({"A": [1.0, 2.0, 3.0], "Flat": [2.0, 2.0, 2.0]}, y=[1.0] * 3)
    corr = dv.pairwise_correlations(wide, ["A", "Flat"]).iloc[0]
    assert np.isnan(corr["pearson_within_series"])


def test_mean_abs_difference_fractions():
    wide = _wide({"A": [2.0, 2.0], "B": [4.0, 4.0]}, y=[2.0, 2.0])
    row = dv.mean_abs_differences(wide, ["A", "B"]).iloc[0]
    assert row["mean_abs_diff"] == pytest.approx(2.0)
    assert row["frac_of_mean_demand"] == pytest.approx(1.0)
    # A's MAE is 0, B's is 2 -> pair MAE 1 -> models disagree by twice what they're wrong by
    assert row["frac_of_pair_mae"] == pytest.approx(2.0)


def test_close_fractions_all_vs_any():
    # series 0: A=B=C totals (all close); series 1: A=B close, C far (any but not all)
    a = [10.0, 10.0, 10.0, 10.0]
    b = [10.0, 10.0, 10.1, 10.1]
    c = [10.0, 10.0, 30.0, 30.0]
    wide = _wide({"A": a, "B": b, "C": c}, y=[1.0] * 4, n_series=2)
    per_pair, summary = dv.close_fractions(wide, ["A", "B", "C"])
    assert summary["all_pairs_close"] == 0.5
    assert summary["any_pair_close"] == 1.0
    ab = per_pair[(per_pair["model_a"] == "A") & (per_pair["model_b"] == "B")]
    assert ab["frac_series_within_5pct"].iloc[0] == 1.0


def test_wide_p50_joins_actuals_and_drops_incomplete_rows():
    rows = []
    for model, p50 in [("A", 1.0), ("B", 2.0)]:
        rows.append(
            {
                "fold": 1,
                "model": model,
                "series_id": "S0",
                "date": pd.Timestamp("2020-01-01"),
                "p50": p50,
                "y": 3.0,
                "zero_rate": 0.2,
            }
        )
    # a row only model A produced — must not survive the pivot
    rows.append({**rows[0], "date": pd.Timestamp("2020-01-02")})
    wide = dv.wide_p50(pd.DataFrame(rows))
    assert len(wide) == 1
    assert list(wide.iloc[0][["A", "B", "y"]]) == [1.0, 2.0, 3.0]
