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

from reorderpoint import calibration as cal
from reorderpoint import decision as dec
from reorderpoint.calibration import SafetyStockCalibration
from reorderpoint.config import REPO_ROOT, Config, load_config
from reorderpoint.exog import REFERENCE_WINDOW_DAYS, future_exog_from_trailing_window  # noqa: F401
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


@lru_cache(maxsize=1)
def _load_calibration() -> SafetyStockCalibration:
    return cal.load()


def get_calibration() -> SafetyStockCalibration:
    """Missing calibration is a 503, exactly like a missing model. There is deliberately no
    fallback to raw quantile-derived sizing: that is the method Phase 4 showed loses, and serving
    it silently is how this system ended up doing so."""
    try:
        calibration = _load_calibration()
        model = _load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # The residuals are one model's; sizing and bands drawn from another's are wrong without
    # looking wrong. Models that do not name themselves are not checked.
    served = getattr(model, "name", None)
    if served is not None and served != calibration.model_name:
        raise HTTPException(
            status_code=409,
            detail=(
                f"the calibration was fitted for {calibration.model_name} but the served model "
                f"is {served}; rerun `make train`"
            ),
        )
    return calibration


def size_decisions(
    preds: pd.DataFrame,
    on_hand: pd.Series,
    config: Config,
    calibration: SafetyStockCalibration,
) -> pd.DataFrame:
    """The one place served reorder points are computed (`/reorder`, `make score`, the
    dashboard): calibrated safety stock through `decision.compute_decisions`, the same sizing rule
    the backtest harness applies (`decision.calibrated_reorder_decision`)."""
    costs = config.costs
    if calibration.lead_time_days != costs.lead_time_days:
        raise ValueError(
            f"calibration was fitted for a {calibration.lead_time_days}-day lead time but the "
            f"config says {costs.lead_time_days}; rerun `make calibrate`"
        )
    return dec.compute_decisions(
        preds,
        on_hand,
        costs.lead_time_days,
        costs.service_level_target,
        safety_stock=calibration.safety_stock(costs.service_level_target, series_ids=on_hand.index),
        lot_multiple=costs.lot_multiple,
    )


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
    calibration: SafetyStockCalibration = Depends(get_calibration),
) -> list[dict]:
    """P50 from the model; `p10`/`p90` from the safety-stock calibration, not the model — the same
    residual calibration `/reorder` sizes with, so the band and the reorder point agree."""
    _validate_series_ids(panel, req.series_ids)
    future_exog = future_exog_from_trailing_window(panel, req.series_ids, req.horizon)
    preds = model.predict_quantiles(req.horizon, future_exog=future_exog)
    preds = calibration.calibrated_interval(preds)
    preds = preds[preds["series_id"].isin(req.series_ids)].sort_values(["series_id", "date"])
    preds["date"] = preds["date"].dt.strftime("%Y-%m-%d")
    return preds.to_dict("records")


@app.post("/reorder")
def reorder(
    req: ReorderRequest,
    model: QuantileForecaster = Depends(get_model),
    panel: pd.DataFrame = Depends(get_history),
    config: Config = Depends(get_config),
    calibration: SafetyStockCalibration = Depends(get_calibration),
) -> list[dict]:
    _validate_series_ids(panel, req.series_ids)
    lead_time_days = config.costs.lead_time_days
    future_exog = future_exog_from_trailing_window(panel, req.series_ids, lead_time_days)
    preds = model.predict_quantiles(lead_time_days, future_exog=future_exog)

    on_hand = pd.Series({sid: req.on_hand.get(sid, 0.0) for sid in req.series_ids})
    try:
        decisions = size_decisions(preds, on_hand, config, calibration)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
    panel: pd.DataFrame = Depends(get_history),
    config: Config = Depends(get_config),
    calibration: SafetyStockCalibration = Depends(get_calibration),
) -> dict:
    return {
        "safety_stock_calibration": calibration.summary(),
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
    calibration = get_calibration()
    series_ids = panel["series_id"].unique().tolist()

    lead_time_days = config.costs.lead_time_days
    future_exog = future_exog_from_trailing_window(panel, series_ids, lead_time_days)
    preds = model.predict_quantiles(lead_time_days, future_exog=future_exog)

    # No live on-hand feed for Track A yet — 0.0 is the only sensible default absent real
    # inventory data (documented limitation, same spirit as docs/data.md's other proxies).
    on_hand = pd.Series(0.0, index=series_ids)
    decisions = size_decisions(preds, on_hand, config, calibration)
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
