#!/bin/bash
#SBATCH --job-name=sm_subset
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h100,vision-shared-h200,vision-shared-a100,vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Flips and side rolls under both contact models, for the clamped runs' iteration-24000 checkpoints.
set -uo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1
mkdir -p results/subset
for run in bones_2gpu_c bones_1gpu_c; do
  for cm in paper legacy; do
    uv run --frozen python tools/eval_subset.py --ckpt logs/$run/ckpt_024000.pt --clips "^(flip|side_roll)" --contact_model $cm \
      --out results/subset/${run}_24000_${cm}.json 2>&1 | grep -v -E "^Module|warn|nefc|Warp|CUDA|Devices|\"cpu\"|\"cuda|Kernel|/data/"
  done
done
