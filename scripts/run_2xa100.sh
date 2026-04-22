#!/usr/bin/env bash
# Minimal, reliable BraTS DynUNet training on 2x A100.
#
# This launcher stays deliberately close to the MONAI BraTS tutorial defaults.
# No custom CUDA kernels, no torch.compile, no channels_last, no fused Adam,
# no deep supervision — just the plain MONAI path, wrapped in DDP.
#
# Enabled:
#   * DDP over NVLink (torchrun, 2 processes, one per GPU)
#   * FP16 autocast + GradScaler
#   * MONAI CacheDataset with cache_rate=1.0  (deterministic transforms cached in RAM)
#   * Standard MONAI DiceLoss, standard Adam, standard memory format
#   * Cosine-annealing LR, cudnn.benchmark, sharded validation
#
# Optional via env vars (override at invocation):
#   WANDB_API_KEY    (required for online logging)
#   DATA_DIR         BraTS root (auto-computed if unset)
#   MAX_EPOCHS       300
#   ROI_SIZE         "128 128 128"
#   NUM_SAMPLES      4
#   BATCH_SIZE       1
#   VAL_INTERVAL     5

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
: "${MAX_EPOCHS:=300}"
: "${ROI_SIZE:=128 128 128}"
: "${NUM_SAMPLES:=4}"
: "${BATCH_SIZE:=1}"
: "${NUM_WORKERS:=4}"
: "${CACHE_RATE:=1.0}"
: "${CACHE_NUM_WORKERS:=8}"
: "${VAL_SW_BATCH:=4}"
: "${VAL_INTERVAL:=5}"
: "${LR:=1e-4}"
export DATA_DIR WANDB_MODE

RUN_NAME="${RUN_NAME:-dynunet-2xa100-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${WORKDIR}/res/models/${RUN_NAME}"
mkdir -p "${LOG_DIR}" "${LOG_DIR}/ranks"

echo "== BraTS DynUNet, 2x A100, simple config =="
echo "Workdir     : ${WORKDIR}"
echo "Data dir    : ${DATA_DIR}"
echo "Run name    : ${RUN_NAME}"
echo "Log dir     : ${LOG_DIR}"
echo "ROI         : ${ROI_SIZE}"
echo "Batch/sample: ${BATCH_SIZE} / ${NUM_SAMPLES}"
echo "Epochs      : ${MAX_EPOCHS}"
echo "GPUs        : ${NPROC_PER_NODE}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "== syncing python environment =="
uv sync --frozen

ulimit -n 1048576

echo "== GPU visibility =="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# NCCL for single-node NVLink
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=1800
export NCCL_TIMEOUT=1800

export OMP_NUM_THREADS="${NUM_WORKERS}"
export MKL_NUM_THREADS="${NUM_WORKERS}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# If the run dies silently, PYTHONFAULTHANDLER prints a Python traceback for
# segfaults / aborts, and torchrun writes per-rank logs under ranks/.
export PYTHONFAULTHANDLER=1

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
    --amp-dtype fp16 \
    --no-deep-supervision \
    --cache-rate "${CACHE_RATE}" \
    --cache-num-workers "${CACHE_NUM_WORKERS}" \
    --no-compile \
    --memory-format standard \
    --no-fused-optimizer \
    --no-fused-dice-loss \
    --no-deterministic \
    --wandb-mode "${WANDB_MODE}" \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
