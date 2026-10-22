#!/bin/bash
#SBATCH --job-name=sm_bones_eval
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Scores every checkpoint of logs/RUN on all filtered BONES-SEED motions (paper App. F.1) as they appear.
# Usage: sbatch slurm/eval_bones.sbatch RUN
set -euo pipefail
RUN=$1
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1
mkdir -p logs/$RUN
uv run --frozen python supermimic/evaluate.py --run_dir logs/$RUN --motion_dir data/bones_seed --watch >> logs/$RUN/eval.log 2>&1
