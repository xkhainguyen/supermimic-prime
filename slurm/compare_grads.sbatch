#!/bin/bash
#SBATCH --job-name=sm_grads
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h100,vision-shared-h200,vision-shared-a100,vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# WM vs simulator Jacobians with finite differences in mjlab (one world per perturbed rollout).
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1 MUJOCO_GL=disable
uv run --frozen python tools/compare_grads.py --backend mjlab --eps 1e-1 1e-2 1e-3 1e-4 --ref_eps 1e-3 --out results/grad_compare_mjlab.json "$@" 2>&1 | grep -v -i -E "warn|run_backward|^Module"
