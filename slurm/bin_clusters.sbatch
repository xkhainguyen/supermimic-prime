#!/bin/bash
#SBATCH --job-name=sm_bin_clusters
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Off-paper cluster-balanced reset prior (allmimic #28) for the BONES-SEED library.
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}"
uv run --frozen python tools/bin_clusters.py --motion_dir data/bones_seed --out data/bin_prior_bones.npz
