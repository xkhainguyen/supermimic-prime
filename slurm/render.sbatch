#!/bin/bash
#SBATCH --job-name=sm_render
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-l40s
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Usage: sbatch slurm/render.sbatch CKPT OUT.mp4 [render.py args...]
# Re-runs the deterministic evaluation of CKPT on all 40 clips with trajectory recording, then renders
# on the GPU (EGL). ffmpeg comes from imageio-ffmpeg; compute nodes have none.
set -euo pipefail
CKPT=$1
OUT=$2
shift 2
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1 MUJOCO_GL=egl
TRAJ=${CKPT%.pt}_traj.npz
[ -f "$TRAJ" ] || uv run --frozen python supermimic/evaluate.py --run_dir "$(dirname "$CKPT")" --ckpt "$CKPT" --record "$TRAJ"
uv run --frozen python tools/render.py "$TRAJ" "$OUT" "$@"
