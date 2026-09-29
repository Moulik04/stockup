FROM python:3.13-slim AS base

# LightGBM's OpenMP runtime — same libomp dependency the README flags for macOS dev (Homebrew),
# here as the Debian package (`libgomp1`) instead. The served model no longer uses LightGBM, but
# `decision.py` imports `backtest.py`, which registers the whole model ladder, so the package still
# imports lightgbm and statsforecast at load; dropping this needs that import split first.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY reorderpoint ./reorderpoint
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# The image serves a pre-trained production model — `make train` writes
# models/production/model.joblib on the host (or in a build stage of your own) before
# building; this image does not bundle the M5 dataset or refit on start (see docs/serving.md).
COPY models/production/model.joblib ./models/production/model.joblib
# The fitted safety-stock calibration `/reorder` sizes with (written by `make train` /
# `make calibrate`). Without it the API returns 503 rather than falling back to the raw
# quantile-derived sizing Phase 4 showed loses — see docs/serving.md.
COPY models/production/safety_stock_calibration.joblib ./models/production/safety_stock_calibration.joblib
COPY data/track_a/processed/panel.parquet ./data/track_a/processed/panel.parquet

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "reorderpoint.serve:app", "--host", "0.0.0.0", "--port", "8000"]
