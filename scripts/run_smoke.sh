#!/usr/bin/env bash
# Absolute-minimum smoke test: single GPU, FP32, no cache, no W&B, no DDP.
#
# The goal of this script is to prove the training loop runs at all. Every
# optional feature is disabled. If this completes one epoch, the model,
# data pipeline, and checkpointing are all working, and the bug in the
# full launcher is in one of: DDP, AMP, CacheDataset, or W&B.
#
# It writes everything — stdout AND stderr — into res/models/<run>/train.log
# so there can be no "silent crash" mystery.
#
# Usage:
#   bash scripts/run_smoke.sh

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

: "${DATA_DIR:=${WORKDIR}/res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"
export DATA_DIR

RUN_NAME="smoke-$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${WORKDIR}/res/models/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

echo "== smoke test =="
echo "Workdir : ${WORKDIR}"
echo "Data dir: ${DATA_DIR}"
echo "Run name: ${RUN_NAME}"
echo "Log     : ${LOG_DIR}/train.log"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv sync --frozen

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.free --format=csv

# Force single GPU, single process. No torchrun, no DDP.
export CUDA_VISIBLE_DEVICES=0
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
# Make any CUDA error point at the line that caused it.
export CUDA_LAUNCH_BLOCKING=1

echo "== launching single-GPU smoke run =="
set -x
uv run python -m src.training.train \
  --model dynunet \
  --data-dir "${DATA_DIR}" \
  --run-name "${RUN_NAME}" \
  --save-dir "${WORKDIR}/res/models" \
  --max-epochs 2 \
  --batch-size 1 \
  --num-samples 2 \
  --roi-size 96 96 96 \
  --num-workers 0 \
  --train-micro-batch-size 1 \
  --val-sw-batch-size 1 \
  --val-interval 1 \
  --lr 1e-4 \
  --no-amp \
  --no-deep-supervision \
  --cache-rate 0.0 \
  --no-compile \
  --memory-format standard \
  --no-fused-optimizer \
  --no-fused-dice-loss \
  --no-deterministic \
  --wandb-mode disabled \
  2>&1 | tee "${LOG_DIR}/train.log"
set +x

echo "== smoke test finished with exit code $? =="
