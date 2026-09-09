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


def test_build_forecast_chart_data_indexes_by_date():
    forecast = pd.DataFrame(
        {
            "series_id": ["A", "A"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "p10": [1.0, 2.0],
            "p50": [2.0, 3.0],
            "p90": [3.0, 4.0],
        }
    )
    out = dash.build_forecast_chart_data(forecast)
    assert list(out.columns) == ["p10", "p50", "p90"]
    assert out.index.name == "date"


def test_latest_report_picks_most_recent_by_filename(tmp_path):
    (tmp_path / "backtest_2026-09-01.md").write_text("old")
    (tmp_path / "backtest_2026-09-05.md").write_text("new")
    (tmp_path / "decision_2026-09-05.md").write_text("unrelated")

    result = dash.latest_report(tmp_path, "backtest")
    assert result == tmp_path / "backtest_2026-09-05.md"


def test_latest_report_returns_none_when_missing(tmp_path):
    assert dash.latest_report(tmp_path, "backtest") is None
