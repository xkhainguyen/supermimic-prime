#!/bin/bash
#SBATCH --job-name=sm_prof
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h100,vision-shared-h200,vision-shared-a100,vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Per-phase timing (SM_PROFILE=1) for BONES-SEED and LAFAN1 on the same GPU, 700 iterations each.
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1 SM_PROFILE=1 TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
nvidia-smi --query-gpu=name --format=csv,noheader
for lib in bones_seed lafan1_g1; do
  echo "=== $lib"
  rm -rf logs/prof_$lib
  uv run --frozen python supermimic/train.py --log_dir logs/prof_$lib --motion_dir data/$lib --iterations 700 --ckpt_every 100000 --log_every 50 2>&1 | grep -E "^\[prof\]|^it (650|700)"
done
