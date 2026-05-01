#!/bin/bash
# Canonical DynUNet batch script.
# Prepare environment on login node first:
#   uv sync --frozen
# Submit with requested GPU tier:
#   sbatch --account=<account> --constraint=gpu40g scripts/train_dynnet.sbatch
#   sbatch --account=<account> --constraint=gpu80g scripts/train_dynnet.sbatch

#SBATCH --partition=GPUQ
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="dynnet"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu40g
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x-%j.out
#SBATCH --error=slurm_outputs/%x-%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

strip_cr() {
  printf '%s' "${1//$'\r'/}"
}

sanitize_var() {
  local var_name="$1"
  printf -v "${var_name}" '%s' "$(strip_cr "${!var_name}")"
}

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKDIR}"
mkdir -p slurm_outputs

echo "Running from: ${WORKDIR}"
echo "Job name: ${SLURM_JOB_NAME:-dynnet}"
echo "Job id: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local-node}"
echo "Constraint: ${SLURM_JOB_CONSTRAINTS:-${SLURM_CONSTRAINT:-unset}}"

if [ ! -x "${WORKDIR}/.venv/bin/python" ]; then
  echo "Missing ${WORKDIR}/.venv/bin/python. Create environment on login node first."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing uv on PATH. Install uv or load the module before submitting this job."
  exit 1
fi

if [ -f "${WORKDIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${WORKDIR}/.env"
  set +a
fi

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET_SOURCE="${DATASET_SOURCE:-cancervision_binary_seg}"
GPU_PROFILE="${GPU_PROFILE:-gpu40g}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-cancervision}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
VAL_SW_BATCH_SIZE="${VAL_SW_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
LOSS="${LOSS:-dicece}"
DICECE_LAMBDA_DICE="${DICECE_LAMBDA_DICE:-0.7}"
DICECE_LAMBDA_CE="${DICECE_LAMBDA_CE:-0.3}"
CROP_POS_WEIGHT="${CROP_POS_WEIGHT:-3}"
CROP_NEG_WEIGHT="${CROP_NEG_WEIGHT:-1}"
VAL_THRESHOLDS="${VAL_THRESHOLDS:-0.30 0.35 0.40 0.45 0.50 0.55 0.60}"
MIN_EPOCHS="${MIN_EPOCHS:-50}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
RUN_NAME="${RUN_NAME:-dynnet-${DATASET_SOURCE}-${SLURM_JOB_ID:-local}}"
TASK_MANIFEST="${TASK_MANIFEST:-${WORKDIR}/res/dataset/cancervision-standardized/task_manifests/segmentation_binary_curated.csv}"
DATA_DIR="${DATA_DIR:-${WORKDIR}/res/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"
ROI_SIZE="${ROI_SIZE:-}"
MODEL_FILTERS="${MODEL_FILTERS:-}"
RESUME="${RESUME:-}"
PATH_PREFIX_MAP="${PATH_PREFIX_MAP:-}"

for var_name in \
  DATASET_SOURCE \
  GPU_PROFILE \
  WANDB_MODE \
  WANDB_PROJECT \
  WANDB_ENTITY \
  MAX_EPOCHS \
  BATCH_SIZE \
  TRAIN_MICRO_BATCH_SIZE \
  NUM_SAMPLES \
  VAL_SW_BATCH_SIZE \
  NUM_WORKERS \
  VAL_INTERVAL \
  LR \
  WEIGHT_DECAY \
  LOSS \
  DICECE_LAMBDA_DICE \
  DICECE_LAMBDA_CE \
  CROP_POS_WEIGHT \
  CROP_NEG_WEIGHT \
  VAL_THRESHOLDS \
  MIN_EPOCHS \
  EARLY_STOP_PATIENCE \
  RUN_NAME \
  TASK_MANIFEST \
  DATA_DIR \
  ROI_SIZE \
  MODEL_FILTERS \
  RESUME \
  PATH_PREFIX_MAP; do
  sanitize_var "${var_name}"
done

echo "Python runner: uv run --no-sync python"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU profile: ${GPU_PROFILE}"
echo "Dataset source: ${DATASET_SOURCE}"
echo "W&B mode: ${WANDB_MODE}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B entity: ${WANDB_ENTITY:-<default>}"
echo "Train micro-batch size: ${TRAIN_MICRO_BATCH_SIZE}"
echo "Validation sw_batch_size: ${VAL_SW_BATCH_SIZE}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Loss: ${LOSS}"
echo "DiceCE lambdas: dice=${DICECE_LAMBDA_DICE} ce=${DICECE_LAMBDA_CE}"
echo "Crop weights: pos=${CROP_POS_WEIGHT} neg=${CROP_NEG_WEIGHT}"
echo "Val thresholds: ${VAL_THRESHOLDS:-<default>}"
echo "Early stop: min_epochs=${MIN_EPOCHS} patience=${EARLY_STOP_PATIENCE}"

nvidia-smi

TRAIN_ARGS=(
  --dataset-source "${DATASET_SOURCE}"
  --gpu-profile "${GPU_PROFILE}"
  --batch-size "${BATCH_SIZE}"
  --train-micro-batch-size "${TRAIN_MICRO_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-samples "${NUM_SAMPLES}"
  --val-sw-batch-size "${VAL_SW_BATCH_SIZE}"
  --max-epochs "${MAX_EPOCHS}"
  --val-interval "${VAL_INTERVAL}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --loss "${LOSS}"
  --dicece-lambda-dice "${DICECE_LAMBDA_DICE}"
  --dicece-lambda-ce "${DICECE_LAMBDA_CE}"
  --crop-pos-weight "${CROP_POS_WEIGHT}"
  --crop-neg-weight "${CROP_NEG_WEIGHT}"
  --min-epochs "${MIN_EPOCHS}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --wandb-mode "${WANDB_MODE}"
  --run-name "${RUN_NAME}"
)

if [ "${DATASET_SOURCE}" = "brats" ]; then
  TRAIN_ARGS+=(--data-dir "${DATA_DIR}")
else
  TRAIN_ARGS+=(--task-manifest "${TASK_MANIFEST}")
fi

if [ -n "${ROI_SIZE}" ]; then
  read -r -a ROI_SIZE_ARGS <<< "${ROI_SIZE}"
  TRAIN_ARGS+=(--roi-size "${ROI_SIZE_ARGS[@]}")
fi

if [ -n "${MODEL_FILTERS}" ]; then
  read -r -a MODEL_FILTER_ARGS <<< "${MODEL_FILTERS}"
  TRAIN_ARGS+=(--model-filters "${MODEL_FILTER_ARGS[@]}")
fi

if [ -n "${VAL_THRESHOLDS}" ]; then
  read -r -a VAL_THRESHOLD_ARGS <<< "${VAL_THRESHOLDS}"
  TRAIN_ARGS+=(--val-thresholds "${VAL_THRESHOLD_ARGS[@]}")
fi

if [ -n "${RESUME}" ]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi

if [ -n "${PATH_PREFIX_MAP}" ]; then
  TRAIN_ARGS+=(--path-prefix-map "${PATH_PREFIX_MAP}")
fi

srun --ntasks=1 --gpus-per-task=1 \
  uv run --no-sync python -m src.models.dynnet \
  "${TRAIN_ARGS[@]}"
