#!/bin/bash
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 3D"
#SBATCH --time=14:15:00
#SBATCH --partition=GPUQ
#SBATCH --job-name="dynunet-brats"
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_terminal.out
#SBATCH --error=slurm_terminal.err
#SBATCH --mail-user=eldarja@ntnu.no
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-.}
cd "${WORKDIR}"
echo "Running from this directory: ${WORKDIR}"
echo "Name of job: ${SLURM_JOB_NAME:-dynunet-brats}"
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

if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
  GPU_COUNT="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
  IFS=',' read -r -a STEP_GPU_ARRAY <<< "${SLURM_STEP_GPUS}"
  GPU_COUNT="${#STEP_GPU_ARRAY[@]}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_VISIBLE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT="${#CUDA_VISIBLE_ARRAY[@]}"
else
  GPU_COUNT=1
fi

echo "Detected GPU count: ${GPU_COUNT}"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "Python: ${WORKDIR}/.venv/bin/python"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

WANDB_MODE_CLEAN="${WANDB_MODE:-online}"
WANDB_MODE_CLEAN="${WANDB_MODE_CLEAN//$'\r'/}"
WANDB_MODE_CLEAN="${WANDB_MODE_CLEAN//$'\n'/}"

nvidia-smi

srun --ntasks="${GPU_COUNT}" --ntasks-per-node="${GPU_COUNT}" --gpus-per-task=1 \
  "${WORKDIR}/.venv/bin/python" -m src.training.train \
  --model dynunet \
  --batch-size 1 \
  --num-samples 4 \
  --roi-size 128 128 128 \
  --wandb-mode "${WANDB_MODE_CLEAN}" \
  --run-name "dynunet-brats-${SLURM_JOB_ID:-local}" \
  --data-dir "${DATA_DIR:-${WORKDIR}/res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"

