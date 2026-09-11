"""N-BEATS via neuralforecast — Phase 7 stretch: a deep model on Bridges-2's V100 GPUs.

Confirmed against the real cluster before writing this — not guessed from library docs (see
docs/bridges2.md, scripts/bridges2/diagnose_nbeats.py): fp16 ("16-mixed") trains correctly on a
V100 (no bf16, no FlashAttention — V100/Volta doesn't support either properly), and
neuralforecast's predict() output columns are `{name}-lo-80.0` / `{name}-median` / `{name}-hi-80.0`
for `MQLoss(quantiles=[0.1, 0.5, 0.9])` — Nixtla's shared lo/hi/level convention (same family as
the statsforecast wrapper in models/base.py), not a `-q-10/50/90` naming.

Unlike the recursive LightGBM rollout (models/lightgbm_global.py), NBEATS forecasts the full
horizon directly in one `predict()` call — no feeding predictions back in as pseudo-history. It's
also univariate here: no calendar/price/category exogenous features, unlike LightGBM's feature
pipeline — a deliberate Phase 7 scoping choice, not an oversight.

This module is only ever imported on Bridges-2 (never by backtest.py directly) — `neuralforecast`
and `torch` aren't installed on the M2. See scripts/bridges2/run_deep_backtest.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NBEATS as _NBEATS

MODEL_NAME = "NBEATS"


class NBEATSModel:
    def __init__(self, horizon: int):
        self._horizon = horizon
        self._nf: NeuralForecast | None = None

    def fit(self, train: pd.DataFrame) -> None:
        df = train.rename(columns={"series_id": "unique_id", "date": "ds"})[
            ["unique_id", "ds", "y"]
        ]
        model = _NBEATS(
            h=self._horizon,
            input_size=2 * self._horizon,
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",
        )
        self._nf = NeuralForecast(models=[model], freq="D")
        self._nf.fit(df=df)

    def predict_quantiles(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("call fit() before predict_quantiles()")
        # neuralforecast's predict() always returns exactly the h it was fit with — unlike
        # LightGBMGlobalModel's recursive rollout, it can't natively serve a different horizon
        # per call. serve.py/dashboard.py call predict_quantiles twice on the same cached model
        # instance with genuinely different horizons (the main forecast, and lead_time_days for
        # the reorder decision) — silently ignoring `horizon` here would return the wrong-length
        # forecast for whichever call didn't match fit time. Serve a shorter request by
        # truncating; fail clearly on a longer one rather than return the wrong length.
        if horizon > self._horizon:
            raise ValueError(
                f"NBEATSModel was fit with horizon={self._horizon}; cannot serve "
                f"predict_quantiles(horizon={horizon}) without refitting."
            )

        forecast = self._nf.predict()
        out = forecast.rename(columns={"unique_id": "series_id", "ds": "date"})[
            ["series_id", "date"]
        ].copy()

        # demand can't be negative; enforce p10 <= p50 <= p90 in case the joint quantile heads
        # cross (the same known quantile-regression artifact handled in lightgbm_global.py).
        p50 = np.clip(forecast[f"{MODEL_NAME}-median"].to_numpy(), 0, None)
        p10 = np.clip(np.minimum(forecast[f"{MODEL_NAME}-lo-80.0"].to_numpy(), p50), 0, None)
        p90 = np.clip(np.maximum(forecast[f"{MODEL_NAME}-hi-80.0"].to_numpy(), p50), 0, None)
        out["p10"] = p10
        out["p50"] = p50
        out["p90"] = p90

        if horizon < self._horizon:
            out = out.sort_values(["series_id", "date"])
            out = out.groupby("series_id", group_keys=False).head(horizon)
        return out


def nbeats_global(horizon: int) -> NBEATSModel:
    return NBEATSModel(horizon)
