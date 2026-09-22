"""Trains the production LightGBM model on the full ingested panel and persists it via joblib —
serve.py loads this instead of retraining on every request/container start.
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from reorderpoint import calibration as cal
from reorderpoint import safety_stock as ss
from reorderpoint.config import REPO_ROOT, load_config
from reorderpoint.models.base import QuantileForecaster
from reorderpoint.models.lightgbm_global import lightgbm_global

PANEL_PATH = REPO_ROOT / "data" / "track_a" / "processed" / "panel.parquet"
MODEL_PATH = REPO_ROOT / "models" / "production" / "lightgbm.joblib"
TRAIN_HORIZON = 28


def train_production_model(panel: pd.DataFrame, horizon: int = TRAIN_HORIZON) -> QuantileForecaster:
    """Fits one LightGBM global model on the entire panel — no held-out fold, since this is the
    model that actually ships (Phase 3's backtest already established it beats the baselines;
    this just trains it on all available history rather than a train/test split).
    """
    model = lightgbm_global(horizon)
    model.fit(panel)
    return model


def train_production_calibration(
    panel: pd.DataFrame, lead_time_days: int
) -> cal.SafetyStockCalibration:
    """The safety-stock calibration served alongside the model — see `calibration.py`. Fits a
    second copy of the model on the panel minus its last lead-time window; the served model is
    still the one trained on everything."""
    return cal.fit_calibration(
        panel,
        lambda: lightgbm_global(TRAIN_HORIZON),
        lead_time_days,
        form=ss.DEFAULT_SCHEME[0],
        granularity=ss.DEFAULT_SCHEME[1],
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
