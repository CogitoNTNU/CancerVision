#!/usr/bin/env bash
# BraTS DynUNet training — 2x A100 DDP, tuned for a ~6h budget.
#
# Philosophy: still reliability-first. No cache, no torch.compile, no fused
# kernels, no deep supervision, no channels_last. The only aggressive choice
# is DDP itself — which is why only well-trodden paths sit around it.
#
# Enabled:
#   * DDP over NVLink (torchrun, 2 processes, one per GPU)
#   * BF16 autocast (no GradScaler, full dynamic range on A100/H100)
#   * AdamW optimizer (decoupled weight decay)
#   * MONAI DiceLoss, cosine-annealing LR fitted to MAX_EPOCHS, cudnn.benchmark
#   * Lazy Dataset (no RAM cache, keeps memory headroom for DDP)
#   * Per-rank stdout/stderr under res/models/<run>/ranks/ for postmortem
#
# Usage:
#   bash scripts/run_2xa100.sh                    # 140-epoch DDP run (~6h)
#   bash scripts/run_2xa100.sh --max-epochs 80    # shorter, ~3.5h
#
# Env overrides:
#   WANDB_API_KEY   required for online logging
#   DATA_DIR        BraTS root
#   NPROC_PER_NODE  2
#   MAX_EPOCHS      140
#   ROI_SIZE        "128 128 128"
#   NUM_SAMPLES     4
#   NUM_WORKERS     2
#   VAL_INTERVAL    5
#   LR              3e-4
#   WEIGHT_DECAY    1e-4

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
: "${NPROC_PER_NODE:=2}"
: "${MAX_EPOCHS:=140}"
: "${ROI_SIZE:=128 128 128}"
: "${NUM_SAMPLES:=4}"
: "${BATCH_SIZE:=1}"
: "${NUM_WORKERS:=2}"
: "${VAL_SW_BATCH:=4}"
: "${VAL_INTERVAL:=5}"
: "${LR:=3e-4}"
: "${WEIGHT_DECAY:=1e-4}"
export DATA_DIR WANDB_MODE

RUN_NAME="${RUN_NAME:-dynunet-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${WORKDIR}/res/models/${RUN_NAME}"
mkdir -p "${LOG_DIR}" "${LOG_DIR}/ranks"

echo "== BraTS DynUNet (2x A100 DDP, AdamW, reliability-first) =="
echo "Workdir    : ${WORKDIR}"
echo "Data dir   : ${DATA_DIR}"
echo "Run name   : ${RUN_NAME}"
echo "Log        : ${LOG_DIR}/train.log"
echo "Ranks dir  : ${LOG_DIR}/ranks"
echo "ROI        : ${ROI_SIZE}"
echo "Epochs     : ${MAX_EPOCHS}"
echo "GPUs       : ${NPROC_PER_NODE}"
echo "LR / WD    : ${LR} / ${WEIGHT_DECAY}"
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

# NCCL over NVLink for single-node 2-GPU. Generous timeouts so a slow first
# epoch (preflight, cache warmup) doesn't get killed by the default 10-min wait.
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=1800
export NCCL_TIMEOUT=1800
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

echo "== preflight =="
uv run python -m src.pipeline \
  --preflight-only \
  --data-dir "${DATA_DIR}" \
  --wandb-mode "${WANDB_MODE}"

echo "== launching DDP training =="
exec uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --log_dir "${LOG_DIR}/ranks" \
  --redirects=3 \
  --tee=3 \
  -m src.training.train \
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
    --weight-decay "${WEIGHT_DECAY}" \
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
