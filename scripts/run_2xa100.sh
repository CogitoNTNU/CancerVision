#!/usr/bin/env bash
# Best-performance BraTS DynUNet (nnU-Net) training on 2x A100 SXM.
#
# Launches DDP across both GPUs via torchrun, enables deep supervision,
# nnU-Net-style augmentations, AMP, cudnn.benchmark, and a larger patch
# size. Defaults assume A100 80GB. On 40GB cards, drop --roi-size to
# 160 160 128 and --num-samples to 2.
#
# Usage:
#   bash scripts/run_2xa100.sh                      # full run (300 epochs)
#   bash scripts/run_2xa100.sh --max-epochs 500     # forward any flag
#
# Env vars via .env or the shell:
#   WANDB_API_KEY  required for online W&B logging
#   DATA_DIR       overrides BraTS root

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
: "${MICRO_BATCH:=2}"
: "${VAL_SW_BATCH:=4}"
: "${LR:=3e-4}"
: "${DEEP_SUPR_NUM:=3}"
export DATA_DIR WANDB_MODE

RUN_NAME="${RUN_NAME:-dynunet-nnunet-2xa100-$(date -u +%Y%m%d-%H%M%S)}"

echo "== 2x A100 nnU-Net run =="
echo "Workdir   : ${WORKDIR}"
echo "Data dir  : ${DATA_DIR}"
echo "Run name  : ${RUN_NAME}"
echo "ROI       : ${ROI_SIZE}"
echo "Epochs    : ${MAX_EPOCHS}"
echo "GPUs      : ${NPROC_PER_NODE}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "== syncing python environment =="
uv sync --frozen

ulimit -n 1048576 || ulimit -n 65536 || true
echo "ulimit -n : $(ulimit -n)"

echo "== GPU visibility =="
nvidia-smi

# NCCL tuning for single-node multi-GPU (SXM NVLink)
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS="${NUM_WORKERS}"
export MKL_NUM_THREADS="${NUM_WORKERS}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "== preflight =="
uv run python -m src.pipeline --skip-preflight --help >/dev/null  # ensures imports ok
uv run python -c "
from pathlib import Path
from src.pipeline import preflight
preflight(Path('${DATA_DIR}'), '${WANDB_MODE}', max_missing_patients=0)
"

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
    --lr "${LR}" \
    --amp \
    --deep-supervision \
    --deep-supr-num "${DEEP_SUPR_NUM}" \
    --no-deterministic \
    --val-interval 2 \
    --wandb-mode "${WANDB_MODE}" \
    "$@"
