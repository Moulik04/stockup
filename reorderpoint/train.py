"""Trains the production LightGBM model on the full ingested panel and persists it via joblib —
serve.py loads this instead of retraining on every request/container start.
"""

from __future__ import annotations

import joblib
import pandas as pd

from reorderpoint.config import REPO_ROOT
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


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    model = train_production_model(panel)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
