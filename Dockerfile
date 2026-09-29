# The Python minor is the one in `.python-version`, which is also what local development and CI
# run. `tests/test_python_pin.py` fails if this default drifts from it, and CI builds with
# `--build-arg PYTHON_VERSION=$(cat .python-version)`. Wheel availability differs between minors
# (statsforecast 2.0.1 has none for 3.13), so the image must not choose its own.
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim AS base

# No OpenMP runtime (`libgomp1`), no compiler: the image installs only the `serve` dependency group
# (pyproject.toml) — what `uvicorn reorderpoint.serve:app` imports, which excludes lightgbm,
# statsforecast and numba (`tests/test_serve_imports.py`). The full dependency set does not build
# in this image: statsforecast ships no Python 3.13 wheel and needs a C++ compiler to build.
#
# uv uses this image's Python instead of downloading its own (pyproject.toml sets
# `python-preference = "only-managed"` for development), so the pinned base image is the one the
# wheels are chosen for, and a `.python-version` that does not match it fails the build loudly.
ENV UV_PYTHON_PREFERENCE=only-system UV_PYTHON_DOWNLOADS=never
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
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
