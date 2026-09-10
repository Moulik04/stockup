#!/bin/bash
# Idempotent environment setup for Phase 7 (deep models) on PSC Bridges-2.
#
# The `pytorch/26.05-2.11-py3` module is a read-only shared uv-managed env (confirmed: `uv pip
# install` into it fails with EACCES). Its own instructions say to clone it into a writable venv
# via `cd $LOCAL; uv venv ...` — but $LOCAL is node-local scratch, wiped after the job ends, so
# that would mean rebuilding this whole environment (torch+cuda included) on every single job
# submission. This script does the same clone, but into persistent storage
# (/ocean/projects/$PSC_ALLOCATION/$USER, confirmed writable, 10GB allocation) instead, so setup
# happens once and every later `sbatch` submission just activates the existing venv.
#
# Safe to re-run: checks the existing venv's own python binary directly (never sources a
# possibly-broken activate script during the check) and does nothing if it's already complete.
# `uv`'s path is captured once right after module load and used explicitly everywhere, so it
# can't go missing later regardless of what sourcing activate/deactivate does to PATH.
#
# Run from a Bridges-2 login node (via the OnDemand web shell or ssh):
#   bash scripts/bridges2/setup_env.sh

set -euo pipefail

VENV_DIR=/ocean/projects/$PSC_ALLOCATION/$USER/stockup-env
CACHE_DIR=/ocean/projects/$PSC_ALLOCATION/$USER/uv_cache
PY_INTERP=/opt/packages/uv/python/cpython-3.13.7-linux-x86_64-gnu

module load pytorch/26.05-2.11-py3
UV_BIN="$(command -v uv)"
echo "Using uv at: $UV_BIN"

verify() {
    # $1 = path to a python3 executable to test (never relies on an activated shell).
    "$1" - <<'PYEOF'
import importlib.util
import sys

# pandas/torch come from the base module; neuralforecast/dotenv/statsforecast/lightgbm are
# this project's own additions (backtest.py/decision.py import all of models/{naive,statistical,
# lightgbm_global} unconditionally, even when a script only uses naive + NBEATS).
required = ["pandas", "torch", "neuralforecast", "dotenv", "statsforecast", "lightgbm"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:", ", ".join(missing))
    sys.exit(1)
print("All required packages present:", ", ".join(required))
PYEOF
}

mkdir -p "$VENV_DIR"
cd "$VENV_DIR"

if [ -x .venv/bin/python3 ] && verify .venv/bin/python3; then
    echo "Environment already complete. Nothing to do."
    echo "Activate in future sessions/jobs with:"
    echo "  module load pytorch/26.05-2.11-py3"
    echo "  source $VENV_DIR/.venv/bin/activate"
    exit 0
fi

echo "Venv missing or incomplete — rebuilding from scratch."
rm -rf .venv pyproject.toml uv.lock

"$UV_BIN" venv --python "$PY_INTERP"

cp /opt/packages/AI/pytorch_26.05-py3/pyproject.toml .
cp /opt/packages/AI/pytorch_26.05-py3/uv.lock .
"$UV_BIN" sync --cache-dir "$CACHE_DIR" --frozen

echo "Adding this project's own packages (one combined install, not several separate calls —"
echo "multiple uv pip install calls in sequence risk the resolver silently dropping something"
echo "already installed when reconciling a later call)..."
"$UV_BIN" pip install --cache-dir "$CACHE_DIR" neuralforecast python-dotenv statsforecast lightgbm

echo ""
echo "Verifying final environment..."
if ! verify .venv/bin/python3; then
    echo "Setup FAILED — see MISSING above. Do not proceed to sbatch until this passes." >&2
    exit 1
fi

echo ""
echo "Setup complete and verified. Activate in future sessions/jobs with:"
echo "  module load pytorch/26.05-2.11-py3"
echo "  source $VENV_DIR/.venv/bin/activate"
