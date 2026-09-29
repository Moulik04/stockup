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


def test_per_series_correlation_is_not_carried_by_the_series_with_big_swings():
    # Series 0 swings by 100 and the two models agree; series 1 swings by 1 and they disagree.
    # The pooled-demeaned column is dominated by series 0; the per-series mean must not be.
    a = [0.0, 100.0, 0.0, 100.0, 1.0, 2.0, 1.0, 2.0]
    b = [0.0, 100.0, 0.0, 100.0, 2.0, 1.0, 2.0, 1.0]
    wide = _wide({"A": a, "B": b}, y=[1.0] * 8, n_series=2)

    corr = dv.pairwise_correlations(wide, ["A", "B"]).iloc[0]
    assert corr["pearson_within_series"] > 0.99
    assert corr["mean_series_pearson"] == pytest.approx(0.0)
    assert corr["frac_series_defined"] == 1.0


def test_per_series_correlation_averages_only_where_both_forecasts_move():
    # series 0: A moves, B is flat (undefined); series 1: both move and agree
    a = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    b = [5.0, 5.0, 5.0, 2.0, 4.0, 6.0]
    wide = _wide({"A": a, "B": b}, y=[1.0] * 6, n_series=2)

    corr = dv.pairwise_correlations(wide, ["A", "B"]).iloc[0]
    assert corr["frac_series_defined"] == 0.5
    assert corr["mean_series_pearson"] == pytest.approx(1.0)


def _forecast_rows(model: str, fold: int, series: str, p50: list[float], y: list[float]):
    dates = pd.date_range("2026-01-01", periods=len(p50))
    return pd.DataFrame(
        {
            "model": model,
            "fold": fold,
            "series_id": series,
            "date": dates,
            "p10": 0.0,
            "p50": p50,
            "p90": 1.0,
            "y": y,
            "zero_rate": 0.5,
        }
    )


def test_lead_time_totals_sum_consecutive_windows_and_drop_a_partial_tail():
    # 9 days with a 4-day lead time: two full windows, the ninth day is dropped
    p50 = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 99.0]
    y = [0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 50.0]
    f = _forecast_rows("A", 1, "s1", p50, y)
    totals = dv.lead_time_totals(f, ["A"], lead_time_days=4)
    assert totals["A"].tolist() == [4.0, 8.0]
    assert totals["y"].tolist() == [6.0, 4.0]
    assert list(totals.index.names) == ["fold", "series_id", "block"]


def test_a_periodic_shape_cancels_over_a_window_the_length_of_its_period():
    weekly = [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0] * 4  # period 7, total 6 per week
    flat = [6.0 / 7] * 28
    y = [1.0] * 28
    f = pd.concat(
        [
            _forecast_rows("Periodic", 1, "s1", weekly, y),
            _forecast_rows("Flat", 1, "s1", flat, y),
        ]
    )
    models = ["Periodic", "Flat"]
    totals = dv.lead_time_totals(f, models, lead_time_days=7)
    assert np.allclose(totals["Periodic"], totals["Flat"])  # identical once summed over the week
    row = dv.lead_time_comparison(dv.wide_p50(f), totals, models).iloc[0]
    assert row["daily_diff_frac"] > 0.5  # they disagree day by day
    assert row["lt_diff_frac"] == pytest.approx(0.0, abs=1e-9)  # and not at all per lead time


def _comparison_row(model_a: str, model_b: str, lt_within: float) -> dict:
    return {
        "model_a": model_a,
        "model_b": model_b,
        "pooled_daily": 0.9,
        "pooled_lt": 0.95,
        "daily_within": 0.5,
        "daily_defined": 0.5,
        "lt_within": lt_within,
        "lt_defined": 1.0,
        "daily_diff_frac": 0.5,
        "lt_diff_frac": 0.2,
        "lt_diff_frac_of_mae": 0.4,
    }


def test_lead_time_verdict_only_claims_agreement_when_within_series_reaches_the_line():
    def verdict(lt_within: float) -> str:
        table = pd.DataFrame(
            [
                _comparison_row("SeasonalNaive", "AutoETS", 0.4),
                _comparison_row("AutoETS", "AutoTheta", lt_within),
            ]
        )
        return "\n".join(dv._lead_time_verdict(table, lead_time_days=7))

    assert "do not converge" in verdict(0.6)
    assert "not supported" in verdict(0.6)
    assert "agree on lead-time totals" in verdict(0.99)
    assert "not supported" not in verdict(0.99)
