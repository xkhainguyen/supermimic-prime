#!/bin/bash
#SBATCH --job-name=sm_lafan
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Usage: sbatch slurm/train.sbatch [RUN_NAME] [ITERATIONS] [train.py args...]
# Trains in the foreground and scores every checkpoint on all 40 clips with an eval watcher on the
# same GPU (logs/<run>/eval.csv). Requeue-safe: resumes from logs/<run>/last.pt (the replay buffer
# is refilled, not restored).
set -euo pipefail
RUN=${1:-lafan1_main}
ITERS=${2:-20000}
shift $(( $# < 2 ? $# : 2 ))
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1
export TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
mkdir -p logs/$RUN
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --frozen python supermimic/evaluate.py --run_dir logs/$RUN --watch >> logs/$RUN/eval.log 2>&1 &
EVAL_PID=$!
trap 'kill $EVAL_PID 2>/dev/null || true' EXIT

uv run --frozen python supermimic/train.py --log_dir logs/$RUN --iterations $ITERS --resume auto "$@" 2>&1 | tee -a logs/$RUN/stdout.log

# score the checkpoints the watcher has not reached yet
kill $EVAL_PID 2>/dev/null || true
uv run --frozen python supermimic/evaluate.py --run_dir logs/$RUN >> logs/$RUN/eval.log 2>&1
