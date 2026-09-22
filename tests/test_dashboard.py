"""Tests for dashboard.py's pure helpers. The actual `st.*` rendering in main() is UI-only and
verified by launching Streamlit directly (see docs/serving.md), not by pytest — same split as
serve.py's route handlers (thin) vs. testable logic.
"""

import pandas as pd
import pytest

from reorderpoint import dashboard as dash
from reorderpoint import holding_breakeven as hb
from reorderpoint.config import annual_to_daily


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


def test_latest_report_ignores_deep_variant_even_though_it_sorts_later(tmp_path):
    # Regression test: a wildcard glob (backtest_*.md) also matches backtest_deep_*.md (the
    # narrower Phase 7 NBEATS-only report), and "deep" sorts lexicographically after any 4-digit
    # year -- confirmed picking the wrong report on a real dashboard screenshot before this fix.
    (tmp_path / "backtest_2026-09-10.md").write_text("comprehensive")
    (tmp_path / "backtest_deep_2026-09-10.md").write_text("nbeats only")

    result = dash.latest_report(tmp_path, "backtest")
    assert result == tmp_path / "backtest_2026-09-10.md"


def test_latest_report_returns_none_when_missing(tmp_path):
    assert dash.latest_report(tmp_path, "backtest") is None


# --- holding-rate slider and the cost-vs-rate chart -------------------------------------------


def _components() -> pd.DataFrame:
    """LightGBM holds less and stocks out more than SeasonalNaive; they cross at 0.001/day."""
    return pd.DataFrame(
        {
            "holding_per_rate": [175_000.0, 350_000.0],
            "stockout": [350.0, 175.0],
        },
        index=[hb.MODEL_A, hb.MODEL_B],
    )


def test_rate_options_always_contain_the_crossover_and_the_configured_rate():
    opts = dash.rate_options(crossover_pct=363.85, configured_pct=14.6)
    assert 364.0 in opts and 14.6 in opts
    assert opts == sorted(opts)
    assert dash.rate_options(None, 25.0).count(25.0) == 1  # no duplicate when already a stop


def test_format_rate_option_tags_the_stops_that_matter():
    assert dash.format_rate_option(364.0, 363.85, 25.0) == "364%/yr · crossover"
    assert dash.format_rate_option(25.0, 363.85, 25.0) == "25%/yr · configured"
    assert dash.format_rate_option(730.0, 363.85, 25.0) == "730%/yr · original default"
    assert dash.format_rate_option(50.0, 363.85, 25.0) == "50%/yr"


def test_cost_verdict_flips_at_the_crossover():
    comp = _components()
    low = dash.cost_verdict(comp, annual_to_daily(0.25))
    high = dash.cost_verdict(comp, annual_to_daily(7.30))
    tie = dash.cost_verdict(comp, 0.001)
    assert f"**{hb.MODEL_B}** is cheaper" in low  # SeasonalNaive cheaper at a plausible rate
    assert f"**{hb.MODEL_A}** is cheaper" in high
    assert "crossover" in tie and "cheaper" not in tie


def test_cost_chart_has_both_lines_the_selected_markers_and_the_crossover():
    comp = _components()
    fig = dash.build_cost_vs_rate_chart(comp, selected_pct=25.0)
    lines = [t for t in fig.data if t.mode == "lines"]
    markers = [t for t in fig.data if t.mode == "markers"]
    assert [t.name for t in lines] == [hb.MODEL_A, hb.MODEL_B]
    # each marker sits on its own line at the selected rate: cost = holding_per_rate * r + stockout
    r = annual_to_daily(0.25)
    assert markers[0].y[0] == pytest.approx(175_000 * r + 350)
    assert markers[1].y[0] == pytest.approx(350_000 * r + 175)
    assert fig.layout.showlegend is True  # two series: a legend is always present
    xs = [s.x0 for s in fig.layout.shapes if s.type == "line"]
    assert pytest.approx(0.001 * 365 * 100) in xs  # the crossover line
    assert 25.0 in xs  # and the selected rate


def test_margin_options_include_the_configured_margin_and_stay_sorted():
    opts = dash.margin_options(27.5)
    assert 27.5 in opts and 24.2 in opts and opts == sorted(opts)
    assert 17.3 in dash.margin_options(17.3)  # a Track B margin not on the default stops
