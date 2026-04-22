#!/usr/bin/env bash
# BraTS DynUNet training on 2x A100 SXM — conservative, known-working stack.
#
# Philosophy: only enable optimisations whose failure modes we've already
# validated end-to-end. Aggressive features (torch.compile, channels_last_3d,
# FusedDiceLoss, fused Adam) are available behind flags but OFF by default
# because they interact badly with DynUNet + deep supervision + MONAI MetaTensor
# on current PyTorch (either hangs in compile, or OOMs on first forward).
#
# Enabled by default:
#   * DDP over NVLink (torchrun, 2 processes)
#   * BF16 autocast (no GradScaler, A100 Tensor Cores, full dynamic range)
#   * Deep supervision (nnU-Net quality, 4 heads)
#   * nnU-Net-style augmentations (noise/smooth/contrast/intensity/flips)
#   * CacheDataset cache_rate=1.0 (deterministic transforms cached in RAM)
#   * Sharded validation across ranks
#   * cudnn.benchmark for fastest conv kernels on fixed ROI shapes
#
# Opt-in experimental flags (can be appended to this invocation):
#   --compile                     torch.compile the model (currently unstable with DS+MONAI)
#   --memory-format channels_last NDHWC convs (can OOM at ROI 192+ with DS)
#   --fused-optimizer             torch.optim.Adam(fused=True)
#   --fused-dice-loss             custom CUDA Dice kernel from src/kernels/
#
# Env overrides (in .env or shell):
#   WANDB_API_KEY   required for online logging
#   DATA_DIR        BraTS root (defaults to res/data/brats/...)
#   AMP_DTYPE       "bf16" (default) or "fp16"
#   ROI_SIZE        "160 160 128" (default) — bump to "192 192 128" if you have headroom

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
: "${ROI_SIZE:=160 160 128}"
: "${NUM_SAMPLES:=2}"
: "${BATCH_SIZE:=2}"
: "${NUM_WORKERS:=8}"
: "${CACHE_NUM_WORKERS:=8}"
: "${CACHE_RATE:=1.0}"
: "${MICRO_BATCH:=2}"
: "${VAL_SW_BATCH:=4}"
: "${VAL_INTERVAL:=5}"
: "${LR:=3e-4}"
: "${DEEP_SUPR_NUM:=3}"
: "${AMP_DTYPE:=bf16}"
export DATA_DIR WANDB_MODE

RUN_NAME="${RUN_NAME:-dynunet-2xa100-$(date -u +%Y%m%d-%H%M%S)}"

echo "== 2x A100 DynUNet (stable config) =="
echo "Workdir     : ${WORKDIR}"
echo "Data dir    : ${DATA_DIR}"
echo "Run name    : ${RUN_NAME}"
echo "ROI         : ${ROI_SIZE}"
echo "Epochs      : ${MAX_EPOCHS}"
echo "GPUs        : ${NPROC_PER_NODE}"
echo "AMP dtype   : ${AMP_DTYPE}"
echo "Cache rate  : ${CACHE_RATE} (${CACHE_NUM_WORKERS} workers)"
echo "Deep supr.  : on (num=${DEEP_SUPR_NUM})"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "== syncing python environment =="
uv sync --frozen

ulimit -n 1048576
echo "ulimit -n : $(ulimit -n)"

echo "== GPU visibility =="
nvidia-smi

# NCCL + timeouts
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=1800
export NCCL_TIMEOUT=1800

export OMP_NUM_THREADS="${NUM_WORKERS}"
export MKL_NUM_THREADS="${NUM_WORKERS}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Catch silent Python traces on OOM / C-side crashes in a visible log file.
LOG_DIR="${WORKDIR}/res/models/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"

echo "== preflight =="
uv run python -m src.pipeline \
  --preflight-only \
  --data-dir "${DATA_DIR}" \
  --wandb-mode "${WANDB_MODE}"

echo "== launching DDP training (logs also tee'd to ${LOG_FILE}) =="
exec uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
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
    "$@" 2>&1 | tee "${LOG_FILE}"
