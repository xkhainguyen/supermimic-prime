#!/bin/bash
#SBATCH --job-name=sm_bones_prep
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Builds the packed 50 Hz BONES-SEED library in ~/datasets/bones_seed_g1_50hz (data/bones_seed links to it).
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}"
uv run --frozen python tools/prepare_bones.py --out /data/pulkitag/users/khain/datasets/bones_seed_g1_50hz --workers 32
