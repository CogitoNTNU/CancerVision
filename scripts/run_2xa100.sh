#!/usr/bin/env bash
# BraTS DynUNet training — reliability-first single-GPU config.
#
# This is the smoke-test config that we've verified runs end-to-end, scaled
# up for real training (more epochs, BF16 for speed, W&B online). It runs on
# one of your two A100s. The second GPU is idle for now; once we have a
# trained baseline from this script, multi-GPU can be re-attempted from a
# known-good starting point.
#
# Philosophy: no DDP, no cache, no custom kernels, no torch.compile. Every
# optimisation that's failed silently on this machine is OFF.
#
# Enabled:
#   * Single A100 (CUDA_VISIBLE_DEVICES=0)
#   * BF16 autocast (no GradScaler, full dynamic range on A100/H100)
#   * Lazy Dataset (no RAM cache)
#   * Standard MONAI DiceLoss, standard Adam, standard memory format
#   * Cosine-annealing LR, cudnn.benchmark
#   * CUDA_LAUNCH_BLOCKING=1 so errors point at the right line
#
# Usage:
#   bash scripts/run_2xa100.sh                    # 80-epoch run (~6h on 1xA100)
#   bash scripts/run_2xa100.sh --max-epochs 300   # longer
#
# Env overrides:
#   WANDB_API_KEY   required for online logging
#   DATA_DIR        BraTS root
#   MAX_EPOCHS      80
#   ROI_SIZE        "128 128 128"
#   NUM_SAMPLES     4
#   NUM_WORKERS     2
#   VAL_INTERVAL    5
#   LR              2e-4

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

if [ -f "${WORKDIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${WORKDIR}/.env"
  set +a
fi

: "${DATA_DIR:=${WORKDIR}/res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"
: "${WANDB_MODE:=online}"
: "${MAX_EPOCHS:=80}"
: "${ROI_SIZE:=128 128 128}"
: "${NUM_SAMPLES:=4}"
: "${BATCH_SIZE:=1}"
: "${NUM_WORKERS:=2}"
: "${VAL_SW_BATCH:=4}"
: "${VAL_INTERVAL:=5}"
: "${LR:=2e-4}"
export DATA_DIR WANDB_MODE

RUN_NAME="${RUN_NAME:-dynunet-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${WORKDIR}/res/models/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

echo "== BraTS DynUNet (single GPU, reliability-first) =="
echo "Workdir    : ${WORKDIR}"
echo "Data dir   : ${DATA_DIR}"
echo "Run name   : ${RUN_NAME}"
echo "Log        : ${LOG_DIR}/train.log"
echo "ROI        : ${ROI_SIZE}"
echo "Epochs     : ${MAX_EPOCHS}"
echo "Batch/samp : ${BATCH_SIZE} / ${NUM_SAMPLES}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "== syncing python environment =="
uv sync --frozen

ulimit -n 1048576

echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# Pin to a single GPU. No DDP.
export CUDA_VISIBLE_DEVICES=0
# Run CUDA synchronously so errors surface at the line that caused them.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

echo "== preflight =="
uv run python -m src.pipeline \
  --preflight-only \
  --data-dir "${DATA_DIR}" \
  --wandb-mode "${WANDB_MODE}"

echo "== launching training =="
exec uv run python -m src.training.train \
  --model dynunet \
  --data-dir "${DATA_DIR}" \
  --run-name "${RUN_NAME}" \
  --save-dir "${WORKDIR}/res/models" \
  --max-epochs "${MAX_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-samples ${NUM_SAMPLES} \
  --roi-size ${ROI_SIZE} \
  --num-workers "${NUM_WORKERS}" \
  --train-micro-batch-size 1 \
  --val-sw-batch-size "${VAL_SW_BATCH}" \
  --val-interval "${VAL_INTERVAL}" \
  --lr "${LR}" \
  --amp \
  --amp-dtype bf16 \
  --no-deep-supervision \
  --cache-rate 0.0 \
  --no-compile \
  --memory-format standard \
  --no-fused-optimizer \
  --no-fused-dice-loss \
  --no-deterministic \
  --wandb-mode "${WANDB_MODE}" \
  "$@" 2>&1 | tee "${LOG_DIR}/train.log"
