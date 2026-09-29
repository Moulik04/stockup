"""Fits the production model (`TrailingMeanModel`, the ladder's MovingAverage rung) on the full
ingested panel and persists it via joblib — serve.py loads this instead of refitting on every
request/container start. Also fits the safety-stock calibration served with it (`calibration.py`).
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from reorderpoint import calibration as cal
from reorderpoint import safety_stock as ss
from reorderpoint.config import REPO_ROOT, load_config
from reorderpoint.models.base import QuantileForecaster
from reorderpoint.models.trailing_mean import TrailingMeanModel

PANEL_PATH = REPO_ROOT / "data" / "track_a" / "processed" / "panel.parquet"
MODEL_PATH = REPO_ROOT / "models" / "production" / "model.joblib"


def train_production_model(panel: pd.DataFrame) -> QuantileForecaster:
    """Fits the served model on the entire panel — no held-out fold, since this is the model that
    actually ships. Its cost against the LightGBM it replaced is measured in
    `reports/production_model_*.md`; there is nothing to train beyond a per-series mean."""
    model = TrailingMeanModel()
    model.fit(panel)
    return model


def train_production_calibration(
    panel: pd.DataFrame, lead_time_days: int
) -> cal.SafetyStockCalibration:
    """The safety-stock calibration served alongside the model — see `calibration.py`. Fits a
    second copy of the model on the panel minus its last lead-time window; the served model is
    still the one trained on everything. Must be refitted whenever the served model changes: the
    residuals are that model's."""
    return cal.fit_calibration(
        panel,
        TrailingMeanModel,
        lead_time_days,
        form=ss.DEFAULT_SCHEME[0],
        granularity=ss.DEFAULT_SCHEME[1],
        model_name=TrailingMeanModel.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the production model + calibration")
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="refit only the safety-stock calibration, leaving the persisted model as is",
    )
    args = parser.parse_args()

    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    if not args.calibration_only:
        model = train_production_model(panel)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Wrote {MODEL_PATH}")

    calibration = train_production_calibration(panel, load_config().costs.lead_time_days)
    cal.save(calibration)
    print(f"Wrote {cal.CALIBRATION_PATH}: {calibration.summary()}")


if __name__ == "__main__":
    main()
