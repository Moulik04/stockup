#!/bin/bash
# One-time environment setup for Phase 7 (deep models) on PSC Bridges-2.
#
# The `pytorch/26.05-2.11-py3` module is a read-only shared uv-managed env (confirmed: `uv pip
# install` into it fails with EACCES). Its own instructions say to clone it into a writable venv
# via `cd $LOCAL; uv venv ...` — but $LOCAL is node-local scratch, wiped after the job ends, so
# that would mean rebuilding this whole environment (torch+cuda included) on every single job
# submission. This script does the same clone, but into persistent storage
# (/ocean/projects/$PSC_ALLOCATION/$USER, confirmed writable, 10GB allocation) instead, so setup
# happens once and every later `sbatch` submission just activates the existing venv.
#
# Run this once, interactively, from a Bridges-2 login node (via the OnDemand web shell or ssh):
#   bash scripts/bridges2/setup_env.sh

set -euo pipefail

VENV_DIR=/ocean/projects/$PSC_ALLOCATION/$USER/stockup-env
mkdir -p "$VENV_DIR"
cd "$VENV_DIR"

module load pytorch/26.05-2.11-py3

uv venv --python /opt/packages/uv/python/cpython-3.13.7-linux-x86_64-gnu
source .venv/bin/activate

cp /opt/packages/AI/pytorch_26.05-py3/pyproject.toml .
cp /opt/packages/AI/pytorch_26.05-py3/uv.lock .
uv sync --cache-dir /ocean/projects/$PSC_ALLOCATION/$USER/uv_cache --frozen

uv pip install --cache-dir /ocean/projects/$PSC_ALLOCATION/$USER/uv_cache neuralforecast

echo ""
echo "Setup complete. Activate in future sessions/jobs with:"
echo "  module load pytorch/26.05-2.11-py3"
echo "  source $VENV_DIR/.venv/bin/activate"
