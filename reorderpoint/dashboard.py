"""Streamlit operator dashboard: pick a series, see its reorder decision, forecast fan chart,
history, backtest performance, and monitoring status. Reads the same production artifacts
`serve.py` does directly (not an HTTP client of the API) — see docs/serving.md.

Layout/color design validated as a mockup first (DECISIONS.md, 2026-09-11) — reorder decision is
the first thing shown (not buried below two charts, the original problem this redesign fixes),
and colors match dataviz's validated palette reference (categorical slot 1 blue, the fixed
good/warning/critical status triad), not hand-picked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run reorderpoint/dashboard.py` executes this file as a direct script path, which
# (like `python scripts/x.py`, see DECISIONS.md) does not add the repo root to sys.path the way
# `-m`/`-c` do -- so `import reorderpoint` below would otherwise fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fastapi import HTTPException

from reorderpoint import decision as dec
from reorderpoint import monitor as mon
from reorderpoint.config import REPO_ROOT, load_config
from reorderpoint.serve import future_exog_from_trailing_window, get_history, get_model
from reorderpoint.train import MODEL_PATH

REPORTS_DIR = REPO_ROOT / "reports"
DRIFT_WINDOW_DAYS = 28

# Colors: dataviz skill's validated reference palette (dark-mode values) — categorical slot 1
# (blue) for the forecast series, the fixed status triad for the reorder badge. Not hand-picked.
BLUE = "#3987e5"
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
    """Most recent `reports/<prefix>_<date>.md` by filename (dates sort lexicographically)."""
    candidates = sorted(reports_dir.glob(f"{prefix}_*.md"))
    return candidates[-1] if candidates else None


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
    config = load_config()
    lead_time_days = config.costs.lead_time_days

    series_ids = sorted(panel["series_id"].unique())

    with st.sidebar:
        st.markdown("## \U0001f4e6 Stockup")
        series_id = st.selectbox("Series", series_ids)
        horizon = st.slider("Forecast horizon (days)", 7, 28, lead_time_days)
        on_hand = st.number_input("Current on-hand stock", min_value=0.0, value=0.0, step=1.0)
        st.caption(f"Track A · HOBBIES · {len(series_ids):,} series")

    st.title(series_id)
    st.caption(
        f"Lead time {lead_time_days} days · "
        f"service level {config.costs.service_level_target:.0%}"
    )

    # Reorder decision first — the thing this whole project produces, not the last thing on the
    # page after two charts (see DECISIONS.md, 2026-09-11: the original demo recording never
    # scrolled far enough to show it).
    lead_time_exog = future_exog_from_trailing_window(panel, [series_id], lead_time_days)
    lead_time_preds = model.predict_quantiles(lead_time_days, future_exog=lead_time_exog)
    decisions = dec.compute_decisions(
        lead_time_preds,
        pd.Series({series_id: on_hand}),
        lead_time_days,
        config.costs.service_level_target,
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
