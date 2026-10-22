#!/bin/bash
#SBATCH --job-name=sm_grads_sweep
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h100,vision-shared-h200,vision-shared-a100,vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# WM vs simulator Jacobians (mjlab finite differences) over contact situations and clips.
# Usage: sbatch slurm/grads_sweep.sbatch [CKPT]
set -uo pipefail
CKPT=${1:-logs/lafan1_main/ckpt_016000.pt}
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1 MUJOCO_GL=disable
mkdir -p results/grads
while read -r sel clip start; do
  echo "=== $sel $clip $start"
  uv run --frozen python tools/compare_grads.py --ckpt $CKPT --backend mjlab --eps 1e-1 1e-2 1e-3 1e-4 --ref_eps 1e-3 \
    --select $sel --clip $clip --start_s $start --out results/grads/${sel}_${clip}.json 2>&1 | grep -E "^window|^floor|^wrote|Error|assert"
done <<'LIST'
touchdown walk1_subject1 4.0
touchdown run1_subject2 3.0
touchdown dance2_subject1 3.0
liftoff walk1_subject1 4.0
liftoff run1_subject2 3.0
stance walk1_subject1 1.0
stance dance1_subject1 2.0
flight run1_subject2 2.0
flight sprint1_subject2 2.0
flight jumps1_subject1 2.0
body fallAndGetUp1_subject1 0.5
body fallAndGetUp2_subject2 0.5
LIST
uv run --frozen python tools/summarize_grads.py "results/grads/*.json"
