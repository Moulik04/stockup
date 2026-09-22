"""The fitted safety-stock calibration the served system sizes with.

The backtest harness sizes safety stock from a model's realised forecast residuals, pooled into
buckets (`safety_stock.py`) — not from `z * σ` with σ read off the model's own P10/P90 width, the
method Phase 4 showed loses. Serving used the loser: `compute_decisions` had no calibration input,
because a calibration needs residuals and residuals come from a backtest, which serving does not
run. This is that missing artifact.

`fit_calibration` holds out the last `lead_time_days` of the panel, fits the production model on
everything before, forecasts the held-out window **with the same proxy future-exog the served
forecast uses** (`exog.py` — a calibration on true future exog would size for a forecast better
than the one actually served), and keeps the per-series lead-time-demand residual plus the two
bucketing variables. That is the same procedure `ablate_safety_stock.calibration_residuals` uses
for fold 1 of the harness. The residuals are persisted, not a target-specific buffer, so the same
artifact sizes for any `service_level_target` and any (form, granularity) `safety_stock.py`
supports; `train.py` writes it next to the model.

Limits, stated because they are real: one calibration window (one residual per series, pooled
across ~5.6k series into two buckets), so a bucket's tail is estimated from that one window; and it
is fitted at train time, so it goes stale exactly as the model does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import joblib
import pandas as pd

from reorderpoint import decision as dec
from reorderpoint import safety_stock as ss
from reorderpoint.config import REPO_ROOT
from reorderpoint.exog import future_exog_from_trailing_window
from reorderpoint.models.base import QuantileForecaster

CALIBRATION_PATH = REPO_ROOT / "models" / "production" / "safety_stock_calibration.joblib"
# Calibration is only a per-series buffer estimate; the bootstrap CI `per_series_safety_stock`
# also returns is for the reports, so serving keeps its resamples minimal.
SERVE_N_BOOT = 2


@dataclass
class SafetyStockCalibration:
    form: str
    granularity: str
    lead_time_days: int
    model_name: str
    calibrated_through: pd.Timestamp  # last date of the held-out residual window
    residuals: pd.DataFrame  # index series_id; columns residual, zero_rate, volume

    def safety_stock(self, service_level: float, n_boot: int = SERVE_N_BOOT) -> pd.Series:
        """Per-series safety stock (units of lead-time demand) at `service_level`."""
        r = self.residuals
        per_series, _ = ss.per_series_safety_stock(
            r["residual"],
            r["zero_rate"],
            r["volume"],
            service_level,
            form=self.form,
            granularity=self.granularity,
            n_boot=n_boot,
        )
        return per_series

    def bucket_table(self, service_level: float) -> pd.DataFrame:
        r = self.residuals
        buckets = ss.assign_buckets(r["zero_rate"], r["volume"], self.granularity)
        return ss.bucket_safety_stock(
            r["residual"], buckets, service_level, form=self.form, n_boot=SERVE_N_BOOT
        )

    def summary(self) -> dict:
        return {
            "form": self.form,
            "granularity": self.granularity,
            "model": self.model_name,
            "lead_time_days": self.lead_time_days,
            "calibrated_through": self.calibrated_through.strftime("%Y-%m-%d"),
            "n_series": int(len(self.residuals)),
        }


def fit_calibration(
    panel: pd.DataFrame,
    model_factory: Callable[[], QuantileForecaster],
    lead_time_days: int,
    form: str = ss.DEFAULT_SCHEME[0],
    granularity: str = ss.DEFAULT_SCHEME[1],
    model_name: str = "LightGBM",
) -> SafetyStockCalibration:
    """Residuals from a held-out final lead-time window — see the module docstring."""
    last = panel["date"].max()
    cutoff = last - pd.Timedelta(days=lead_time_days - 1)
    calib_train = panel[panel["date"] < cutoff]
    calib_test = panel[panel["date"] >= cutoff]

    zero_rates, volumes = ss.train_stats(calib_train)
    model = model_factory()
    model.fit(calib_train)
    series_ids = sorted(calib_train["series_id"].unique())
    exog = future_exog_from_trailing_window(calib_train, series_ids, lead_time_days)
    preds = model.predict_quantiles(lead_time_days, future_exog=exog)
    residuals = dec.lead_time_forecast_residuals(
        preds, calib_test[["series_id", "date", "y"]], lead_time_days
    )
    table = pd.DataFrame(
        {
            "residual": residuals,
            "zero_rate": zero_rates.reindex(residuals.index),
            "volume": volumes.reindex(residuals.index),
        }
    )
    table.index.name = "series_id"
    return SafetyStockCalibration(
        form=form,
        granularity=granularity,
        lead_time_days=lead_time_days,
        model_name=model_name,
        calibrated_through=last,
        residuals=table,
    )


def save(calibration: SafetyStockCalibration, path=CALIBRATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibration, path)


def load(path=CALIBRATION_PATH) -> SafetyStockCalibration:
    if not path.exists():
        raise RuntimeError(
            f"no safety-stock calibration at {path} — run `make train` (or `make calibrate`)"
        )
    return joblib.load(path)
