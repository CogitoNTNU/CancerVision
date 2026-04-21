#!/bin/bash
# Canonical Idun DynUNet batch script.
# Prepare the virtual environment on the login node before submitting:
#   uv sync --frozen
# Smoke test submission:
#   sbatch --account=<account> --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 \
#     --mem=32G --time=00:20:00 scripts/train_brats.sbatch

#SBATCH --partition=GPUQ
#SBATCH --account=<account>
#SBATCH --job-name="dynunet-brats"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x-%j.out
#SBATCH --error=slurm_outputs/%x-%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR}"
cd "${WORKDIR}"
mkdir -p slurm_outputs

echo "Running from: ${WORKDIR}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Job id: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE:-unknown}"

module purge
module load CUDA/12.4.0
module load cuDNN/9.1.0-CUDA-12.4.0

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

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "Python: ${WORKDIR}/.venv/bin/python"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

srun "${WORKDIR}/.venv/bin/python" -m src.training.train \
  --model dynunet \
  --run-name "dynunet-brats-${SLURM_JOB_ID}" \
  --wandb-mode "${WANDB_MODE:-offline}" \
  --data-dir "${DATA_DIR:-${WORKDIR}/res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"
