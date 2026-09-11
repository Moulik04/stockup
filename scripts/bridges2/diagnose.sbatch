#!/bin/bash
#SBATCH --job-name=stockup-diag
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

: "${PSC_ALLOCATION:?Set PSC_ALLOCATION to your Bridges-2 project allocation ID first.}"

module load pytorch/26.05-2.11-py3
source "/ocean/projects/${PSC_ALLOCATION}/${USER}/stockup-env/.venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"
python3 scripts/bridges2/diagnose_nbeats.py
