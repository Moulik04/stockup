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

module load pytorch/26.05-2.11-py3
source /ocean/projects/$PSC_ALLOCATION/$USER/stockup-env/.venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

# Same panel Phases 2-6 already used — download + ingest once if not already present.
if [ ! -f data/track_a/processed/panel.parquet ]; then
    echo "panel.parquet not found — run 'make data' (or copy it over) before this job."
    exit 1
fi

python3 scripts/bridges2/run_deep_backtest.py
