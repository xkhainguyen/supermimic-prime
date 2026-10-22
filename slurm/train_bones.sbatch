#!/bin/bash
#SBATCH --job-name=sm_bones
#SBATCH --account=vision-pulkitag
#SBATCH --partition=vision-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/data/pulkitag/users/khain/supermimic-mjlab/logs/slurm/%x_%j.out
# Paper full setup (Table A.5) on the filtered BONES-SEED library, 4096 envs per GPU.
# Usage: sbatch [--gres=gpu:2] slurm/train_bones.sbatch RUN [ITERATIONS] [train.py args...]
# Multi-GPU runs use torchrun (data parallel, lr x sqrt(#GPUs)). Evaluation runs in a separate job
# (slurm/eval_bones.sbatch), so train_wall_h is training only. Requeue-safe via logs/<run>/last.pt.
set -euo pipefail
RUN=${1:-bones_1gpu}
ITERS=${2:-320000}
shift $(( $# < 2 ? $# : 2 ))
cd /data/pulkitag/users/khain/supermimic-mjlab
export PATH="${HOME}/.local/bin:${PATH}" MJLAB_WARP_QUIET=1
export TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/inductor
mkdir -p logs/$RUN
NGPU=$(nvidia-smi -L | wc -l)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ARGS=(--log_dir logs/$RUN --motion_dir data/bones_seed --iterations $ITERS --ckpt_every 4000 --resume auto "$@")
if (( NGPU > 1 )); then
  uv run --frozen torchrun --standalone --nproc_per_node=$NGPU supermimic/train.py "${ARGS[@]}" 2>&1 | tee -a logs/$RUN/stdout.log
else
  uv run --frozen python supermimic/train.py "${ARGS[@]}" 2>&1 | tee -a logs/$RUN/stdout.log
fi
