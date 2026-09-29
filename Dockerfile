FROM python:3.13-slim AS base

# No OpenMP runtime (`libgomp1`), no compiler: the image installs only the `serve` dependency group
# (pyproject.toml) — what `uvicorn reorderpoint.serve:app` imports, which excludes lightgbm,
# statsforecast and numba (`tests/test_serve_imports.py`). The full dependency set does not build
# in this image: statsforecast ships no Python 3.13 wheel and needs a C++ compiler to build.
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group serve

# The project itself is not installed (`--only-group` skips it); the app runs from /app.
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app
COPY reorderpoint ./reorderpoint

# The image serves a pre-trained production model — `make train` writes
# models/production/model.joblib on the host (or in a build stage of your own) before
# building; this image does not bundle the M5 dataset or refit on start (see docs/serving.md).
COPY models/production/model.joblib ./models/production/model.joblib
# The fitted safety-stock calibration `/reorder` and `/forecast` use (written by `make train` /
# `make calibrate`). Without it the API returns 503 rather than falling back to the raw
# quantile-derived sizing Phase 4 showed loses — see docs/serving.md.
COPY models/production/safety_stock_calibration.joblib ./models/production/safety_stock_calibration.joblib
COPY data/track_a/processed/panel.parquet ./data/track_a/processed/panel.parquet

EXPOSE 8000

CMD ["uvicorn", "reorderpoint.serve:app", "--host", "0.0.0.0", "--port", "8000"]
