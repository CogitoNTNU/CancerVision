#!/usr/bin/env bash
# Max-performance BraTS DynUNet (nnU-Net) training on 2x A100 SXM 80GB.
#
# Stack (in the order it helps):
#   * DDP across both GPUs via torchrun (NVLink P2P through NCCL)
#   * CacheDataset  : deterministic transforms cached in RAM, only crops/flips rerun
#   * BF16 autocast : no GradScaler, full dynamic range (A100/H100 sweet spot)
#   * channels_last_3d memory format (faster 3D convs on A100 Tensor Cores)
#   * torch.compile(mode="default") : kernel fusion after fixed ROI shapes
#   * fused Adam (torch.optim.Adam(fused=True))
#   * custom fused Dice CUDA kernel (src/kernels/fused_dice.cu)
#   * Deep supervision + nnU-Net augmentations for quality
#
# Usage:
#   bash scripts/run_2xa100.sh                     # 300-epoch production run
#   bash scripts/run_2xa100.sh --max-epochs 500    # any extra flag is forwarded
#   DATA_DIR=/custom/path bash scripts/run_2xa100.sh
#   AMP_DTYPE=fp16 bash scripts/run_2xa100.sh      # if BF16 ever behaves oddly
#
# On 40GB A100s drop ROI_SIZE to "160 160 128" and NUM_SAMPLES=2.

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
: "${ROI_SIZE:=192 192 128}"
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

RUN_NAME="${RUN_NAME:-dynunet-nnunet-2xa100-$(date -u +%Y%m%d-%H%M%S)}"

echo "== 2x A100 nnU-Net (max perf) =="
echo "Workdir   : ${WORKDIR}"
echo "Data dir  : ${DATA_DIR}"
echo "Run name  : ${RUN_NAME}"
echo "ROI       : ${ROI_SIZE}"
echo "Epochs    : ${MAX_EPOCHS}"
echo "GPUs      : ${NPROC_PER_NODE}"
echo "AMP dtype : ${AMP_DTYPE}"
echo "Cache     : ${CACHE_RATE} (${CACHE_NUM_WORKERS} workers)"

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

# NCCL tuning for single-node NVLink
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Val sliding-window at 192^3 can take several minutes; bump watchdog to 30 min.
export TORCH_NCCL_TIMEOUT=1800
export NCCL_TIMEOUT=1800

export OMP_NUM_THREADS="${NUM_WORKERS}"
export MKL_NUM_THREADS="${NUM_WORKERS}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Cache the JIT-compiled fused Dice kernel in the repo to avoid recompiling
# on re-runs (small, portable between same-CUDA hosts).
export TORCH_EXTENSIONS_DIR="${WORKDIR}/.torch_extensions"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

echo "== preflight (validates data, CUDA, W&B, compiles fused Dice kernel) =="
uv run python -m src.pipeline \
  --preflight-only \
  --data-dir "${DATA_DIR}" \
  --wandb-mode "${WANDB_MODE}" \
  --fused-dice-loss

echo "== launching DDP training =="
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
    --compile \
    --memory-format channels_last \
    --fused-optimizer \
    --fused-dice-loss \
    --no-deterministic \
    --wandb-mode "${WANDB_MODE}" \
    "$@"
