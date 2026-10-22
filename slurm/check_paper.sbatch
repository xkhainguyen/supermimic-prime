#!/bin/bash
#SBATCH --job-name=sm_check
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h100,vision-shared-h200,vision-shared-a100
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Env checks plus a short paper-config BONES-SEED training run with per-phase timing.
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1 TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
nvidia-smi --query-gpu=name --format=csv,noheader
uv run --frozen python tools/test_env.py 2>&1 | grep -v -E "^Module|warn"
rm -rf logs/check_paper
SM_PROFILE=1 uv run --frozen python supermimic/train.py --log_dir logs/check_paper --motion_dir data/bones_seed --iterations 400 --ckpt_every 100000 --log_every 50 2>&1 | grep -E "^\[prof\]|^it |Traceback|Error|overflow|nconmax|njmax" | awk '/prof/{n++; if(n%5!=1) next} {print}'
