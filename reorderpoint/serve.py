"""FastAPI app: POST /forecast, POST /reorder, GET /health, GET /metrics.

The production model and ingested panel are loaded once (cached) rather than per-request; the
panel's trailing exog window stands in for real forward-looking calendar/price data an actual
deployment would source from a promo/pricing system — see `future_exog_from_trailing_window`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from reorderpoint import decision as dec
from reorderpoint.config import REPO_ROOT, Config, load_config
from reorderpoint.models.base import QuantileForecaster
from reorderpoint.train import MODEL_PATH, PANEL_PATH

OUTPUTS_DIR = REPO_ROOT / "outputs"

app = FastAPI(title="ReorderPoint")


class ForecastRequest(BaseModel):
    series_ids: list[str]
    horizon: int = 28


class ReorderRequest(BaseModel):
    series_ids: list[str]
    on_hand: dict[str, float] = {}


def future_exog_from_trailing_window(
    panel: pd.DataFrame, series_ids: list[str], horizon: int
) -> pd.DataFrame:
    """Production stand-in for real forward-looking exog (future calendar/events/price): repeats
    each series' most recent `horizon`-day row pattern, with dates advanced past the panel's last
    known date. Calendar fields that are genuinely knowable in advance (wday/month/year) are
    recomputed from the real future dates rather than carried over stale; everything else (price,
    events, SNAP) is a repeat-the-recent-pattern proxy — a real deployment would source these from
    an actual pricing/promo/calendar system, the same kind of disclosed simplification as
    y-as-demand-proxy in docs/data.md.
    """
    sub = panel[panel["series_id"].isin(series_ids)].sort_values(["series_id", "date"])
    last_date = panel["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    frames = []
    for series_id, group in sub.groupby("series_id"):
        trailing = group.tail(horizon).reset_index(drop=True)
        if trailing.empty:
            continue
        if len(trailing) < horizon:
            reps = horizon // len(trailing) + 1
            trailing = pd.concat([trailing] * reps, ignore_index=True)
        trailing = trailing.iloc[:horizon].copy()
        trailing["date"] = future_dates
        if "wday" in trailing.columns:
            trailing["wday"] = trailing["date"].dt.dayofweek + 1
        if "month" in trailing.columns:
            trailing["month"] = trailing["date"].dt.month
        if "year" in trailing.columns:
            trailing["year"] = trailing["date"].dt.year
        trailing["series_id"] = series_id
        frames.append(trailing.drop(columns=["y"], errors="ignore"))
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def _load_model() -> QuantileForecaster:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"no production model at {MODEL_PATH} — run `make train` first")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def get_model() -> QuantileForecaster:
    try:
        return _load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_history() -> pd.DataFrame:
    return _load_panel()


def get_config() -> Config:
    return load_config()


def _validate_series_ids(panel: pd.DataFrame, series_ids: list[str]) -> None:
    missing = sorted(set(series_ids) - set(panel["series_id"].unique()))
    if missing:
        raise HTTPException(status_code=404, detail=f"unknown series_id(s): {missing}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast")
def forecast(
    req: ForecastRequest,
    model: QuantileForecaster = Depends(get_model),
    panel: pd.DataFrame = Depends(get_history),
) -> list[dict]:
    _validate_series_ids(panel, req.series_ids)
    future_exog = future_exog_from_trailing_window(panel, req.series_ids, req.horizon)
    preds = model.predict_quantiles(req.horizon, future_exog=future_exog)
    preds = preds[preds["series_id"].isin(req.series_ids)].sort_values(["series_id", "date"])
    preds["date"] = preds["date"].dt.strftime("%Y-%m-%d")
    return preds.to_dict("records")


@app.post("/reorder")
def reorder(
    req: ReorderRequest,
    model: QuantileForecaster = Depends(get_model),
    panel: pd.DataFrame = Depends(get_history),
    config: Config = Depends(get_config),
) -> list[dict]:
    _validate_series_ids(panel, req.series_ids)
    lead_time_days = config.costs.lead_time_days
    future_exog = future_exog_from_trailing_window(panel, req.series_ids, lead_time_days)
    preds = model.predict_quantiles(lead_time_days, future_exog=future_exog)

    on_hand = pd.Series({sid: req.on_hand.get(sid, 0.0) for sid in req.series_ids})
    decisions = dec.compute_decisions(
        preds, on_hand, lead_time_days, config.costs.service_level_target
    )
    decisions = decisions.reset_index().rename(columns={"index": "series_id"})
    cols = [
        "series_id",
        "reorder_point",
        "safety_stock",
        "order_quantity",
        "stockout_probability",
        "rationale",
    ]
    return decisions[cols].to_dict("records")


@app.get("/metrics")
def metrics(
    panel: pd.DataFrame = Depends(get_history), config: Config = Depends(get_config)
) -> dict:
    return {
        "n_series": int(panel["series_id"].nunique()),
        "panel_last_date": panel["date"].max().strftime("%Y-%m-%d"),
        "track": config.track,
        "lead_time_days": config.costs.lead_time_days,
        "service_level_target": config.costs.service_level_target,
    }


def run_batch() -> None:
    panel = get_history()
    model = get_model()
    config = get_config()
    series_ids = panel["series_id"].unique().tolist()

    lead_time_days = config.costs.lead_time_days
    future_exog = future_exog_from_trailing_window(panel, series_ids, lead_time_days)
    preds = model.predict_quantiles(lead_time_days, future_exog=future_exog)

    # No live on-hand feed for Track A yet — 0.0 is the only sensible default absent real
    # inventory data (documented limitation, same spirit as docs/data.md's other proxies).
    on_hand = pd.Series(0.0, index=series_ids)
    decisions = dec.compute_decisions(
        preds, on_hand, lead_time_days, config.costs.service_level_target
    )
    decisions = decisions.reset_index().rename(columns={"index": "series_id"})

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = dt.date.today().isoformat()
    out_path = OUTPUTS_DIR / f"reorder_{date_str}.csv"
    decisions.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ReorderPoint serving")
    parser.add_argument("--batch", action="store_true", help="score every series to a CSV")
    args = parser.parse_args()
    if args.batch:
        run_batch()
    else:
        print("Use `make serve` (uvicorn) to run the API, or --batch for batch scoring.")


if __name__ == "__main__":
    main()
