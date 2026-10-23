#!/bin/bash
#SBATCH --job-name=sm_smooth
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h100,vision-shared-h200,vision-shared-a100
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# WM policy gradient vs randomized-smoothing simulator gradients (mjlab finite differences).
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1
uv run --frozen python tools/smooth_compare.py "$@" 2>&1 | grep -v -E "^Module|warn|nefc|Warp|CUDA|Devices|\"cpu\"|\"cuda|Kernel|/data/"
