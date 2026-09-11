#!/bin/bash
#SBATCH --job-name=stockup-nbeats
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

: "${PSC_ALLOCATION:?Set PSC_ALLOCATION to your Bridges-2 project allocation ID first.}"

module load pytorch/26.05-2.11-py3
source "/ocean/projects/${PSC_ALLOCATION}/${USER}/stockup-env/.venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"

# Fail fast rather than burn GPU allocation time on an environment problem. This should never
# actually trigger if `bash scripts/bridges2/setup_env.sh` was run (and passed) beforehand —
# that's the real pre-flight check, since it runs on the login node with no GPU wait. This is a
# belt-and-suspenders check, not a substitute for running setup_env.sh first.
python3 -c "import pandas, torch, neuralforecast, dotenv, statsforecast, lightgbm, scipy" || {
    echo "Environment is missing required packages — run scripts/bridges2/setup_env.sh first." >&2
    exit 1
}

# Same panel Phases 2-6 already used — copied over separately, since it's gitignored.
if [ ! -f data/track_a/processed/panel.parquet ]; then
    echo "panel.parquet not found — copy it over (see docs/bridges2.md) before this job." >&2
    exit 1
fi

python3 scripts/bridges2/run_deep_backtest.py
