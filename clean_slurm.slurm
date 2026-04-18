#!/bin/bash
#SBATCH --account=studiegrupper-cogito
#SBATCH --time=24:00:00
#SBATCH --partition=GPUQ
#SBATCH --job-name="dynunet-cancervision-native"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_outputs/%x-%j.out
#SBATCH --error=slurm_outputs/%x-%j.err
#SBATCH --mail-user=eldarja@ntnu.no
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-.}
cd "${WORKDIR}"
mkdir -p slurm_outputs

echo "Running from this directory: ${WORKDIR}"
echo "Name of job: ${SLURM_JOB_NAME:-dynunet-cancervision-native}"
echo "ID of job: ${SLURM_JOB_ID:-local}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST:-local-node}"

if [ ! -x "${WORKDIR}/.venv/bin/python" ]; then
  echo "Missing ${WORKDIR}/.venv/bin/python. Create the environment on the login node first."
  exit 1
fi

if [ -f "${WORKDIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${WORKDIR}/.env"
  set +a
fi

echo "Single-task launch only"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Python: ${WORKDIR}/.venv/bin/python"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

WANDB_MODE_CLEAN="${WANDB_MODE:-online}"
WANDB_MODE_CLEAN="${WANDB_MODE_CLEAN//$'\r'/}"
WANDB_MODE_CLEAN="${WANDB_MODE_CLEAN//$'\n'/}"
TASK_MANIFEST="${TASK_MANIFEST:-${WORKDIR}/res/dataset/cancervision-standardization/task_manifests/segmentation_binary_curated.csv}"
TASK_MANIFEST="${TASK_MANIFEST//$'\r'/}"
TASK_MANIFEST="${TASK_MANIFEST//$'\n'/}"
GPU_PROFILE_CLEAN="${GPU_PROFILE:-gpu80g}"
BATCH_SIZE_CLEAN="${BATCH_SIZE:-1}"
TRAIN_MICRO_BATCH_SIZE_CLEAN="${TRAIN_MICRO_BATCH_SIZE:-1}"
NUM_SAMPLES_CLEAN="${NUM_SAMPLES:-1}"
VAL_SW_BATCH_SIZE_CLEAN="${VAL_SW_BATCH_SIZE:-1}"
NUM_WORKERS_CLEAN="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
MAX_EPOCHS_CLEAN="${MAX_EPOCHS:-200}"
VAL_INTERVAL_CLEAN="${VAL_INTERVAL:-1}"
LR_CLEAN="${LR:-1e-4}"
WEIGHT_DECAY_CLEAN="${WEIGHT_DECAY:-1e-5}"
ROI_SIZE_CLEAN="${ROI_SIZE:-144 144 144}"
MODEL_FILTERS_CLEAN="${MODEL_FILTERS:-32 64 128 256 320}"

read -r -a ROI_SIZE_ARGS <<< "${ROI_SIZE_CLEAN}"
read -r -a MODEL_FILTER_ARGS <<< "${MODEL_FILTERS_CLEAN}"

nvidia-smi
echo "Requested Slurm constraint: gpu80g"
echo "GPU profile: ${GPU_PROFILE_CLEAN}"
echo "ROI size: ${ROI_SIZE_CLEAN}"
echo "Model filters: ${MODEL_FILTERS_CLEAN}"
echo "Batch size: ${BATCH_SIZE_CLEAN}"
echo "Train micro-batch size: ${TRAIN_MICRO_BATCH_SIZE_CLEAN}"
echo "Num samples: ${NUM_SAMPLES_CLEAN}"
echo "Validation sw_batch_size: ${VAL_SW_BATCH_SIZE_CLEAN}"
echo "Num workers: ${NUM_WORKERS_CLEAN}"
echo "Task manifest: ${TASK_MANIFEST}"

TRAIN_ARGS=(
  --gpu-profile "${GPU_PROFILE_CLEAN}"
  --dataset-source cancervision_binary_seg
  --batch-size "${BATCH_SIZE_CLEAN}"
  --train-micro-batch-size "${TRAIN_MICRO_BATCH_SIZE_CLEAN}"
  --num-workers "${NUM_WORKERS_CLEAN}"
  --num-samples "${NUM_SAMPLES_CLEAN}"
  --val-sw-batch-size "${VAL_SW_BATCH_SIZE_CLEAN}"
  --max-epochs "${MAX_EPOCHS_CLEAN}"
  --val-interval "${VAL_INTERVAL_CLEAN}"
  --lr "${LR_CLEAN}"
  --weight-decay "${WEIGHT_DECAY_CLEAN}"
  --wandb-mode "${WANDB_MODE_CLEAN}"
  --run-name "dynunet-cancervision-native-${SLURM_JOB_ID:-local}"
  --task-manifest "${TASK_MANIFEST}"
  --roi-size "${ROI_SIZE_ARGS[@]}"
  --model-filters "${MODEL_FILTER_ARGS[@]}"
)

srun --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1 \
  "${WORKDIR}/.venv/bin/python" -m src.models.dynnet \
  "${TRAIN_ARGS[@]}"
