"""Tests for dashboard.py's pure helpers. The actual `st.*` rendering in main() is UI-only and
verified by launching Streamlit directly (see docs/serving.md), not by pytest — same split as
serve.py's route handlers (thin) vs. testable logic.
"""

import pandas as pd

from reorderpoint import dashboard as dash


def test_load_series_history_filters_and_trims_to_trailing_window():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    panel = pd.DataFrame(
        {
            "series_id": ["A"] * 10 + ["B"] * 10,
            "date": list(dates) * 2,
            "y": list(range(10)) + list(range(100, 110)),
        }
    )
    out = dash.load_series_history(panel, "A", days=3)
    assert list(out["y"]) == [7, 8, 9]
    assert (out["date"] <= dates[-1]).all()


def test_build_fan_chart_has_band_and_median_traces():
    forecast = pd.DataFrame(
        {
            "series_id": ["A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "p10": [1.0, 2.0],
            "p50": [2.0, 3.0],
            "p90": [3.0, 4.0],
        }
    )
    fig = dash.build_fan_chart(forecast)
    assert len(fig.data) == 2
    band, median = fig.data
    assert band.fill == "toself"
    assert list(median.y) == [2.0, 3.0]


def test_build_history_chart_plots_actuals():
    history = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]), "y": [4.0, 5.0]})
    fig = dash.build_history_chart(history)
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [4.0, 5.0]


def test_status_for_thresholds():
    assert dash.status_for(0.05) == ("healthy", dash.GOOD)
    assert dash.status_for(0.3) == ("watch", dash.WARNING)
    assert dash.status_for(0.8) == ("reorder now", dash.CRITICAL)


def test_latest_report_picks_most_recent_by_filename(tmp_path):
    (tmp_path / "backtest_2026-09-01.md").write_text("old")
    (tmp_path / "backtest_2026-09-05.md").write_text("new")
    (tmp_path / "decision_2026-09-05.md").write_text("unrelated")

    result = dash.latest_report(tmp_path, "backtest")
    assert result == tmp_path / "backtest_2026-09-05.md"


def test_latest_report_returns_none_when_missing(tmp_path):
    assert dash.latest_report(tmp_path, "backtest") is None
