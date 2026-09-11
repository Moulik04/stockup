"""Streamlit operator dashboard: pick a series, see its history, forecast fan chart, reorder
recommendation, backtest performance, and monitoring status. Reads the same production artifacts
`serve.py` does directly (not an HTTP client of the API) — see docs/serving.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run reorderpoint/dashboard.py` executes this file as a direct script path, which
# (like `python scripts/x.py`, see DECISIONS.md) does not add the repo root to sys.path the way
# `-m`/`-c` do -- so `import reorderpoint` below would otherwise fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from fastapi import HTTPException

from reorderpoint import decision as dec
from reorderpoint import monitor as mon
from reorderpoint.config import REPO_ROOT, load_config
from reorderpoint.serve import future_exog_from_trailing_window, get_history, get_model
from reorderpoint.train import MODEL_PATH

REPORTS_DIR = REPO_ROOT / "reports"
DRIFT_WINDOW_DAYS = 28


def load_series_history(panel: pd.DataFrame, series_id: str, days: int = 180) -> pd.DataFrame:
    """Trailing `days` of actual sales for one series, for the history chart."""
    sub = panel[panel["series_id"] == series_id].sort_values("date")
    return sub.tail(days)[["date", "y"]]


def build_forecast_chart_data(forecast: pd.DataFrame) -> pd.DataFrame:
    """Reshapes a p10/p50/p90 forecast frame into st.line_chart's wide, date-indexed format."""
    return forecast.set_index("date")[["p10", "p50", "p90"]]


def latest_report(reports_dir: Path, prefix: str) -> Path | None:
    """Most recent `reports/<prefix>_<date>.md` by filename (dates sort lexicographically)."""
    candidates = sorted(reports_dir.glob(f"{prefix}_*.md"))
    return candidates[-1] if candidates else None


def main() -> None:
    st.set_page_config(page_title="ReorderPoint", layout="wide")
    st.title("ReorderPoint")

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
    series_id = st.selectbox("Series", series_ids)

    st.subheader("History")
    st.line_chart(load_series_history(panel, series_id).set_index("date")["y"])

    st.subheader("Forecast (P10 / P50 / P90)")
    horizon = st.slider("Forecast horizon (days)", 7, 28, lead_time_days)
    future_exog = future_exog_from_trailing_window(panel, [series_id], horizon)
    forecast = model.predict_quantiles(horizon, future_exog=future_exog)
    st.line_chart(build_forecast_chart_data(forecast))

    st.subheader("Reorder recommendation")
    on_hand = st.number_input("Current on-hand stock", min_value=0.0, value=0.0, step=1.0)
    lead_time_exog = future_exog_from_trailing_window(panel, [series_id], lead_time_days)
    lead_time_preds = model.predict_quantiles(lead_time_days, future_exog=lead_time_exog)
    decisions = dec.compute_decisions(
        lead_time_preds,
        pd.Series({series_id: on_hand}),
        lead_time_days,
        config.costs.service_level_target,
    )
    row = decisions.loc[series_id]
    col1, col2, col3 = st.columns(3)
    col1.metric("Reorder point", f"{row['reorder_point']:.0f}")
    col2.metric("Order quantity", f"{row['order_quantity']:.0f}")
    col3.metric("Stockout probability", f"{row['stockout_probability']:.1%}")
    st.caption(row["rationale"])

    st.subheader("Backtest performance")
    for prefix, label in [("backtest", "Accuracy backtest"), ("decision", "Decision cost")]:
        path = latest_report(REPORTS_DIR, prefix)
        with st.expander(f"{label} ({path.name if path else 'no report yet'})"):
            st.markdown(path.read_text() if path else f"Run `make {prefix}` to generate one.")

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
