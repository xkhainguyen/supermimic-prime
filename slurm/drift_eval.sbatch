#!/bin/bash
#SBATCH --job-name=sm_drift
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-pulkitag-l40s,vision-pulkitag-a100,vision-pulkitag-h100,vision-pulkitag-h200
#SBATCH --qos=vision-pulkitag-main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Global-pose drift of checkpoints (tools/drift_eval.py). Usage: sbatch slurm/drift_eval.sbatch "CKPT [CKPT...]" [args...]
set -euo pipefail
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1
CKPTS=$1; shift
for CKPT in $CKPTS; do
  uv run --frozen python tools/drift_eval.py --ckpt $CKPT --out ${CKPT%.pt}_drift.json "$@"
done
