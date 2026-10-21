#!/bin/bash
#SBATCH --job-name=sm_test
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --frozen python tools/test_env.py "$@"
