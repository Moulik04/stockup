"""Streamlit operator dashboard: pick a series, see its reorder decision, forecast fan chart,
history, backtest performance, and monitoring status. Reads the same production artifacts
`serve.py` does directly (not an HTTP client of the API) — see docs/serving.md.

Layout/color design validated as a mockup first — reorder decision is the first thing shown (not
buried below two charts, the original problem this redesign fixes), and colors match dataviz's
validated palette reference (categorical slot 1 blue, the fixed good/warning/critical status
triad), not hand-picked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run reorderpoint/dashboard.py` executes this file as a direct script path, which
# (like `python scripts/x.py`) does not add the repo root to sys.path the way `-m`/`-c` do -- so
# `import reorderpoint` below would otherwise fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fastapi import HTTPException

from reorderpoint import decision as dec
from reorderpoint import holding_breakeven as hb
from reorderpoint import monitor as mon
from reorderpoint.config import (
    DEFAULT_GROSS_MARGIN,
    PLAUSIBLE_ANNUAL_HOLDING_RATE,
    REPO_ROOT,
    annual_to_daily,
    daily_to_annual,
    load_config,
)
from reorderpoint.serve import (
    future_exog_from_trailing_window,
    get_calibration,
    get_history,
    get_model,
    size_decisions,
)
from reorderpoint.train import MODEL_PATH

REPORTS_DIR = REPO_ROOT / "reports"
DRIFT_WINDOW_DAYS = 28

# Colors: dataviz skill's validated reference palette (dark-mode values) — categorical slot 1
# (blue) for the forecast series, the fixed status triad for the reorder badge. Not hand-picked.
BLUE = "#3987e5"
ORANGE = "#d95926"  # categorical slot 2 (dark), validated with slot 1: CVD dE 26.8, normal 31.8
BAND = "rgba(195, 194, 183, 0.14)"  # neutral: shading marks a range, it is not a series
BLUE_BAND = "rgba(57, 135, 229, 0.20)"
HISTORY_INK = "#c3c2b7"
GRID = "#2b2d31"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"


def load_series_history(panel: pd.DataFrame, series_id: str, days: int = 90) -> pd.DataFrame:
    """Trailing `days` of actual sales for one series, for the history chart."""
    sub = panel[panel["series_id"] == series_id].sort_values("date")
    return sub.tail(days)[["date", "y"]]


def latest_report(reports_dir: Path, prefix: str) -> Path | None:
    """Most recent `reports/<prefix>_<date>.md` by filename (dates sort lexicographically).

    The glob is date-shaped (`????-??-??`), not `*` — `backtest_*.md` also matches
    `backtest_deep_2026-09-10.md` (the narrower Phase 7 NBEATS-only report), and "deep" sorts
    lexicographically after any 4-digit year, so a wildcard glob would silently pick that report
    instead of the comprehensive one — confirmed happening in a real dashboard screenshot before
    this fix.
    """
    candidates = sorted(reports_dir.glob(f"{prefix}_????-??-??.md"))
    return candidates[-1] if candidates else None


# Annual holding-cost stops for the slider. Roughly log-spaced: the plausible range (15-30%/yr) and
# the crossover (~364%/yr) are a factor of twelve apart, so a linear slider would make the range a
# sliver. The crossover and the configured rate are added as stops so both can be landed on exactly.
RATE_STOPS_PCT = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 600, 730, 800)
LEGACY_ANNUAL_PCT = 730


def rate_options(crossover_pct: float | None, configured_pct: float) -> list[float]:
    """Slider stops (annual %), with the crossover and the configured rate always among them."""
    extra = {round(configured_pct, 1)} | ({float(round(crossover_pct))} if crossover_pct else set())
    return sorted({float(v) for v in RATE_STOPS_PCT} | extra)


def format_rate_option(pct: float, crossover_pct: float | None, configured_pct: float) -> str:
    tags = []
    if crossover_pct and pct == float(round(crossover_pct)):
        tags.append("crossover")
    if pct == round(configured_pct, 1):
        tags.append("configured")
    if pct == LEGACY_ANNUAL_PCT:
        tags.append("original default")
    label = f"{pct:g}%/yr"
    return f"{label} · {', '.join(tags)}" if tags else label


def cost_verdict(comp: pd.DataFrame, daily_rate: float) -> str:
    """One sentence: who is cheapest at this rate, and how the two headline models compare."""
    costs = hb.cost_at(comp, daily_rate)
    gap = costs[hb.MODEL_A] - costs[hb.MODEL_B]
    if abs(gap) < 1:
        return (
            f"At {daily_to_annual(daily_rate):.0%}/yr {hb.MODEL_A} and {hb.MODEL_B} cost the same "
            f"(${costs[hb.MODEL_A]:,.0f} per fold): this is the crossover."
        )
    cheaper = hb.MODEL_B if gap > 0 else hb.MODEL_A
    return (
        f"At {daily_to_annual(daily_rate):.0%}/yr **{costs.idxmin()}** is the cheapest of the "
        f"{len(costs)} models (${costs.min():,.0f} per fold). Of {hb.MODEL_A} and {hb.MODEL_B}, "
        f"**{cheaper}** is cheaper by ${abs(gap):,.0f}."
    )


def build_cost_vs_rate_chart(
    comp: pd.DataFrame, selected_pct: float, x_max_pct: float = 800.0
) -> go.Figure:
    """Both models' simulated cost against the holding rate: exact straight lines (the policy never
    reads the rate), the plausible range shaded, the crossover and the selected rate marked."""
    xs = np.linspace(0, x_max_pct, 161)
    daily = annual_to_daily(xs / 100)
    crossover = hb.breakeven(comp)
    crossover_pct = daily_to_annual(crossover) * 100 if crossover else None

    fig = go.Figure()
    for model, color in ((hb.MODEL_A, BLUE), (hb.MODEL_B, ORANGE)):
        row = comp.loc[model]
        ys = row["holding_per_rate"] * daily + row["stockout"]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=model,
                line=dict(color=color, width=2),
                hovertemplate="%{x:.0f}%/yr: $%{y:,.0f}" + f"<extra>{model}</extra>",
            )
        )
        here = row["holding_per_rate"] * annual_to_daily(selected_pct / 100) + row["stockout"]
        fig.add_trace(
            go.Scatter(
                x=[selected_pct],
                y=[here],
                mode="markers",
                showlegend=False,
                marker=dict(size=10, color=color, line=dict(color="#1a1a19", width=2)),
                hoverinfo="skip",
            )
        )
    lo, hi = (v * 100 for v in PLAUSIBLE_ANNUAL_HOLDING_RATE)
    fig.add_vrect(
        x0=lo,
        x1=hi,
        fillcolor=BAND,
        line_width=0,
        annotation_text="plausible",
        annotation_position="top left",
        annotation_font_color=HISTORY_INK,
    )
    if crossover_pct:
        fig.add_vline(
            x=crossover_pct,
            line=dict(color=HISTORY_INK, width=1, dash="dot"),
            annotation_text=f"crossover {crossover_pct:.0f}%",
            annotation_position="top",
            annotation_font_color=HISTORY_INK,
        )
    fig.add_vline(x=selected_pct, line=dict(color=HISTORY_INK, width=2))
    _bare_chart_layout(fig, height=320)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.18, x=0),
        margin=dict(l=8, r=8, t=36, b=8),
    )
    fig.update_xaxes(title_text="Holding cost, % of unit cost per year", ticksuffix="%")
    fig.update_yaxes(title_text="Cost per 28-day fold", tickprefix="$")
    return fig


# Lost-sale cost as a share of price (gross margin), for the second slider. The default is Walmart
# U.S.'s 27.5% (config.DEFAULT_GROSS_MARGIN); the range runs to a lost sale costing nearly the whole
# price, which is roughly what the pre-2026-09-20 flat $5 penalty amounted to.
MARGIN_STOPS_PCT = (5, 10, 15, 20, 24.2, 27.5, 30, 40, 50, 60, 85)


def margin_options(configured_pct: float) -> list[float]:
    return sorted({float(v) for v in MARGIN_STOPS_PCT} | {round(configured_pct, 1)})


@st.cache_data(show_spinner=False)
def load_cost_components(mtime: float, gross_margin: float | None) -> pd.DataFrame | None:
    """Per-model holding/stockout split from the stored decision backtest, re-priced per SKU under
    `gross_margin` (`economics.reprice`). `mtime` is only the cache key, so a rerun of
    `make decide` is picked up without restarting Streamlit."""
    try:
        return hb.model_components(pd.read_parquet(dec.DETAIL_PATH), gross_margin=gross_margin)
    except (FileNotFoundError, OSError):
        return None


def status_for(stockout_probability: float) -> tuple[str, str]:
    """(label, color) for the reorder status badge — a 3-tier traffic light on a continuous
    probability, using the dataviz skill's fixed status colors (never repurposed as a 4th
    categorical series elsewhere on the page)."""
    if stockout_probability < 0.15:
        return "healthy", GOOD
    if stockout_probability < 0.5:
        return "watch", WARNING
    return "reorder now", CRITICAL


def _bare_chart_layout(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=HISTORY_INK, size=12, family="IBM Plex Sans, sans-serif"),
        showlegend=False,
        xaxis=dict(gridcolor=GRID, showline=False, zeroline=False),
        yaxis=dict(gridcolor=GRID, showline=False, zeroline=False, rangemode="tozero"),
    )
    return fig


def build_fan_chart(forecast: pd.DataFrame) -> go.Figure:
    """P10-P90 shaded band + P50 line, replacing the old 3-separate-line chart — see dataviz
    skill's mark specs (thin lines, direct end label via hover, recessive gridlines)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["p90"], forecast["p10"][::-1]]),
            fill="toself",
            fillcolor=BLUE_BAND,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="P10–P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["p50"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=4, color=BLUE),
            name="P50",
            hovertemplate="%{x|%b %d}: %{y:.1f}<extra>P50</extra>",
        )
    )
    return _bare_chart_layout(fig)


def build_history_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["y"],
            mode="lines",
            line=dict(color=HISTORY_INK, width=1.5),
            hovertemplate="%{x|%b %d}: %{y:.0f}<extra></extra>",
        )
    )
    return _bare_chart_layout(fig)


_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
)


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{_FONTS_URL}" rel="stylesheet">
        <style>
        html, body {{ font-family: 'IBM Plex Sans', sans-serif; }}
        div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_decision_card(row: pd.Series) -> None:
    label, color = status_for(row["stockout_probability"])
    with st.container(border=True):
        top_left, top_right = st.columns([3, 1])
        top_left.markdown("**Reorder decision**")
        top_right.markdown(
            f'<div style="text-align:right">'
            f'<span style="background:{color}26;color:{color};padding:4px 10px;'
            f'border-radius:100px;font-size:12px;font-weight:600;">● {label}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Reorder point",
            f"{row['reorder_point']:.0f} units",
            help="A demand-driven threshold from the forecast alone — this does NOT change with "
            "on-hand stock, by design. Order quantity and stockout probability do.",
        )
        c2.metric("Order quantity", f"{row['order_quantity']:.0f} units")
        c3.metric("Stockout probability", f"{row['stockout_probability']:.0%}")
        st.caption(row["rationale"])


def main() -> None:
    st.set_page_config(page_title="Stockup", page_icon="\U0001f4e6", layout="wide")
    inject_custom_css()

    if not MODEL_PATH.exists():
        st.error(f"No production model at {MODEL_PATH} — run `make train` first.")
        return

    panel = get_history()
    try:
        model = get_model()
    except HTTPException as exc:
        # get_model() is shared with serve.py's FastAPI dependency injection, where raising
        # HTTPException is correct — but Streamlit doesn't know what to do with it (a TOCTOU
        # race after the MODEL_PATH.exists() check above would otherwise surface as a raw,
        # unhandled exception instead of the same graceful message that check already gives).
        st.error(exc.detail)
        return
    try:
        # the same calibrated sizing `/reorder` uses — never the raw quantile-derived fallback
        calibration = get_calibration()
    except HTTPException as exc:
        st.error(exc.detail)
        return
    config = load_config()
    lead_time_days = config.costs.lead_time_days
    configured_pct = daily_to_annual(config.costs.holding_cost_rate) * 100
    configured_margin_pct = (
        config.costs.gross_margin if config.costs.gross_margin is not None else DEFAULT_GROSS_MARGIN
    ) * 100
    have_backtest = dec.DETAIL_PATH.exists()

    series_ids = sorted(panel["series_id"].unique())

    with st.sidebar:
        st.markdown("## \U0001f4e6 Stockup")
        series_id = st.selectbox("Series", series_ids)
        horizon = st.slider("Forecast horizon (days)", 7, 28, lead_time_days)
        on_hand = st.number_input("Current on-hand stock", min_value=0.0, value=0.0, step=1.0)
        selected_pct = configured_pct
        comp, crossover_pct, can_compare = None, None, False
        if have_backtest:
            st.markdown("**Cost of a lost sale**")
            margin_pct = st.select_slider(
                "% of price (gross margin)",
                options=margin_options(configured_margin_pct),
                value=round(configured_margin_pct, 1),
                format_func=lambda v: f"{v:g}%" + (" · Walmart U.S. FY2026" if v == 27.5 else ""),
                help="What a lost sale costs, as a share of the item's price; stock is valued at "
                "the rest. Which model is cheapest depends on this and on the holding cost.",
            )
            comp = load_cost_components(dec.DETAIL_PATH.stat().st_mtime, margin_pct / 100)
            can_compare = comp is not None and {hb.MODEL_A, hb.MODEL_B} <= set(comp.index)
            crossover = hb.breakeven(comp) if can_compare else None
            crossover_pct = daily_to_annual(crossover) * 100 if crossover else None
        if can_compare:
            st.markdown("**Holding cost**")
            selected_pct = st.select_slider(
                "% of unit cost per year",
                options=rate_options(crossover_pct, configured_pct),
                value=round(configured_pct, 1),
                format_func=lambda v: format_rate_option(v, crossover_pct, configured_pct),
                help="Which model is cheapest depends on this. It does not change the reorder "
                "decision, which is sized from the service level.",
            )
            marks = f"plausible 15–30% · original default {LEGACY_ANNUAL_PCT}%"
            if crossover_pct:
                marks = f"crossover {crossover_pct:.0f}% · " + marks
            st.caption(marks)
        st.caption(f"Track A · HOBBIES · {len(series_ids):,} series")

    st.title(series_id)
    st.caption(
        f"Lead time {lead_time_days} days · "
        f"service level {config.costs.service_level_target:.0%} · "
        f"holding {configured_pct:.0f}%/yr of unit cost"
    )

    # Reorder decision first — the thing this whole project produces, not the last thing on the
    # page after two charts (the original demo recording never scrolled far enough to show it).
    lead_time_exog = future_exog_from_trailing_window(panel, [series_id], lead_time_days)
    lead_time_preds = model.predict_quantiles(lead_time_days, future_exog=lead_time_exog)
    decisions = size_decisions(
        lead_time_preds, pd.Series({series_id: on_hand}), config, calibration
    )
    render_decision_card(decisions.loc[series_id])

    chart_col, history_col = st.columns([1.4, 1])
    with chart_col:
        st.subheader("Forecast")
        future_exog = future_exog_from_trailing_window(panel, [series_id], horizon)
        forecast = model.predict_quantiles(horizon, future_exog=future_exog)
        st.plotly_chart(
            build_fan_chart(forecast), width="stretch", config={"displayModeBar": False}
        )
    with history_col:
        st.subheader("History")
        st.plotly_chart(
            build_history_chart(load_series_history(panel, series_id)),
            width="stretch",
            config={"displayModeBar": False},
        )

    if can_compare:
        st.subheader("Which model is cheapest to run?")
        cost_col, detail_col = st.columns([1.4, 1])
        daily = annual_to_daily(selected_pct / 100)
        with cost_col:
            st.plotly_chart(
                build_cost_vs_rate_chart(comp, selected_pct),
                width="stretch",
                config={"displayModeBar": False},
            )
        with detail_col:
            st.markdown(cost_verdict(comp, daily))
            costs_here = hb.cost_at(comp, daily).sort_values()
            st.dataframe(
                costs_here.rename("cost per fold").map("${:,.0f}".format).to_frame(),
                width="stretch",
            )
            st.caption(
                "The reorder point and order quantity above do not move with these sliders: they "
                "are sized from the service level and the forecast, never from the holding rate "
                "or the lost-sale cost. The two decide which *forecasting model* is cheapest to "
                "run — the verdict above updates with them. Simulated on the Track A backtest at "
                "the default 95% service target; at the cost-optimal target the gap between "
                "LightGBM and SeasonalNaive shrinks to statistical noise. See "
                "reports/holding_breakeven_*.md and reports/penalty_sensitivity_*.md."
            )

    perf_col, monitor_col = st.columns(2)
    with perf_col:
        st.subheader("Backtest performance")
        for prefix, label in [("backtest", "Accuracy backtest"), ("decision", "Decision cost")]:
            path = latest_report(REPORTS_DIR, prefix)
            with st.expander(f"{label} ({path.name if path else 'no report yet'})"):
                st.markdown(path.read_text() if path else f"Run `make {prefix}` to generate one.")

    with monitor_col:
        st.subheader("Monitoring status")
        st.caption(
            "No live forecast log yet (no scheduled job appending real outcomes) — showing a live "
            f"input-drift check on this panel's own history instead: last {DRIFT_WINDOW_DAYS} days "
            "vs. everything before that, per series (reorderpoint.monitor.detect_input_drift)."
        )
        cutoff = panel["date"].max() - pd.Timedelta(days=DRIFT_WINDOW_DAYS)
        baseline = panel[panel["date"] < cutoff]
        recent = panel[panel["date"] >= cutoff]
        drifted = mon.detect_input_drift(baseline, recent)
        if series_id in drifted:
            st.warning(f"{series_id}: recent sales have drifted from its training-window baseline.")
        else:
            st.success(f"{series_id}: no drift flagged.")
        st.caption(f"{len(drifted)}/{len(series_ids)} series flagged across the whole panel.")


if __name__ == "__main__":
    main()
