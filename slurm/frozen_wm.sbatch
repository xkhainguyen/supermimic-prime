#!/bin/bash
#SBATCH --job-name=sm_frozen
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h100,vision-shared-h200,vision-shared-a100,vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Frozen-WM stress test. Usage: sbatch slurm/frozen_wm.sbatch LOG_NAME WM_CKPT [frozen_wm.py args...]
set -euo pipefail
NAME=$1; CKPT=$2; shift 2
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1 TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
mkdir -p logs/stress/$NAME
uv run --frozen python tools/frozen_wm.py --wm_ckpt $CKPT --log_dir logs/stress/$NAME "$@" 2>&1 | grep -v -E "^Module|nefc overflow" | tee logs/stress/$NAME/stdout.log
