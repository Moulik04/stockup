"""One-time diagnostic: confirm neuralforecast's actual NBEATS API and output shape on this
cluster before writing the real integration (reorderpoint/models/deep.py). Run via
diagnose.sbatch on a GPU node — the login node has no GPU, so torch.cuda checks here are
meaningless there.

Prints, rather than assumes:
- neuralforecast version and NBEATS's real constructor signature (docs/memory can drift from
  what's actually installed)
- whether CUDA is visible and which GPU
- the exact column names nf.predict() produces with MQLoss(quantiles=[0.1, 0.5, 0.9])
- whether fp16 ("16-mixed") training actually runs without falling back to bf16 (V100 doesn't
  support bf16 tensor cores properly — the master prompt's fp16-only constraint exists for
  exactly this reason)
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

import neuralforecast  # noqa: E402
from neuralforecast import NeuralForecast  # noqa: E402
from neuralforecast.losses.pytorch import MQLoss  # noqa: E402
from neuralforecast.models import NBEATS  # noqa: E402

print("neuralforecast:", neuralforecast.__version__)
print()
print("NBEATS.__init__ signature:")
print(inspect.signature(NBEATS.__init__))

rng = np.random.default_rng(0)
dates = pd.date_range("2020-01-01", periods=200, freq="D")
rows = [
    {"unique_id": uid, "ds": d, "y": float(i % 10) + rng.random()}
    for uid in ("A", "B", "C")
    for i, d in enumerate(dates)
]
df = pd.DataFrame(rows)

horizon = 28
print()
print(f"Fitting a tiny NBEATS (h={horizon}, max_steps=50) on 3 synthetic series...")

model = NBEATS(
    h=horizon,
    input_size=2 * horizon,
    loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
    max_steps=50,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="16-mixed",
)
nf = NeuralForecast(models=[model], freq="D")
nf.fit(df=df)
forecast = nf.predict()

print()
print("predict() columns:", forecast.columns.tolist())
print(forecast.head(10))
print()
print("DONE")
