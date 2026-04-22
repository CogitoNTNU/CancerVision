#!/usr/bin/env bash
# BraTS DynUNet training on 2x A100 SXM — known-working configuration.
#
# This launcher mirrors exactly the flag set that previously trained past
# the first validation in this repo. No experimental performance flags.
# Per-rank stderr is captured to disk so silent deaths become visible.
#
# Stack:
#   * DDP (torchrun, 2 procs)
#   * FP16 autocast + GradScaler  (proven stable; BF16 is also safe but this
#                                   matches the prior run exactly)
#   * Deep supervision (nnU-Net quality)
#   * CacheDataset cache_rate=1.0
#   * Standard MONAI DiceLoss, standard Adam, standard memory format
#   * cudnn.benchmark, sharded validation
#
# Run it:
#   bash scripts/run_2xa100.sh
#
# Env overrides:
#   WANDB_API_KEY   (required for online logging)
#   DATA_DIR        BraTS root
#   AMP_DTYPE       "fp16" (default) or "bf16"
#   ROI_SIZE        "128 128 128" (default, tutorial-aligned)
#   MAX_EPOCHS      300
#   VAL_INTERVAL    5

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
: "${CACHE_NUM_WORKERS:=8}"
: "${CACHE_RATE:=1.0}"
: "${MICRO_BATCH:=1}"
: "${VAL_SW_BATCH:=4}"
: "${VAL_INTERVAL:=5}"
: "${LR:=3e-4}"
: "${DEEP_SUPR_NUM:=3}"
: "${AMP_DTYPE:=fp16}"
export DATA_DIR WANDB_MODE

RUN_NAME="${RUN_NAME:-dynunet-2xa100-$(date -u +%Y%m%d-%H%M%S)}"
LOG_DIR="${WORKDIR}/res/models/${RUN_NAME}"
mkdir -p "${LOG_DIR}" "${LOG_DIR}/ranks"

echo "== 2x A100 DynUNet =="
echo "Workdir     : ${WORKDIR}"
echo "Data dir    : ${DATA_DIR}"
echo "Run name    : ${RUN_NAME}"
echo "Log dir     : ${LOG_DIR}"
echo "ROI         : ${ROI_SIZE}"
echo "Epochs      : ${MAX_EPOCHS}"
echo "GPUs        : ${NPROC_PER_NODE}"
echo "AMP dtype   : ${AMP_DTYPE}"
echo "Batch/sample: ${BATCH_SIZE}/${NUM_SAMPLES} (micro=${MICRO_BATCH})"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "== syncing python environment =="
uv sync --frozen

ulimit -n 1048576

echo "== GPU visibility =="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# NCCL + timeouts
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=1800
export NCCL_TIMEOUT=1800

# Make silent failures visible
export TORCHELASTIC_ERROR_FILE="${LOG_DIR}/torchelastic_error.json"
export CUDA_LAUNCH_BLOCKING=0   # set to 1 to debug async CUDA errors; slower
export PYTHONFAULTHANDLER=1      # dump Python traceback on SIGSEGV/SIGABRT

export OMP_NUM_THREADS="${NUM_WORKERS}"
export MKL_NUM_THREADS="${NUM_WORKERS}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "== preflight =="
uv run python -m src.pipeline \
  --preflight-only \
  --data-dir "${DATA_DIR}" \
  --wandb-mode "${WANDB_MODE}"

echo "== launching DDP training =="
echo "Rank stderr logs: ${LOG_DIR}/ranks/"
echo "Combined stdout : ${LOG_DIR}/train.log"
echo

# torchrun --redirects=3 sends each rank's stdout/stderr to
# ${LOG_DIR}/ranks/<rank>/stdout.log and stderr.log. The parent still shows
# rank-0's stdout via --tee=3 so the tmux session remains interactive.
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
    --train-micro-batch-size "${MICRO_BATCH}" \
    --val-sw-batch-size "${VAL_SW_BATCH}" \
    --val-interval "${VAL_INTERVAL}" \
    --lr "${LR}" \
    --amp \
    --amp-dtype "${AMP_DTYPE}" \
    --deep-supervision \
    --deep-supr-num "${DEEP_SUPR_NUM}" \
    --cache-rate "${CACHE_RATE}" \
    --cache-num-workers "${CACHE_NUM_WORKERS}" \
    --no-compile \
    --memory-format standard \
    --no-fused-optimizer \
    --no-fused-dice-loss \
    --no-deterministic \
    --wandb-mode "${WANDB_MODE}" \
    "$@" 2>&1 | tee "${LOG_DIR}/train.log"
