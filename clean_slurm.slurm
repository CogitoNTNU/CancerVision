#!/bin/bash
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 3D"
#SBATCH --time=14:15:00
#SBATCH --partition=GPUQ
#SBATCH --job-name="dynunet-brats"
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_terminal.out
#SBATCH --error=slurm_terminal.err
#SBATCH --mail-user=eldarja@ntnu.no
#SBATCH --mail-type=ALL

set -euo pipefail

# p100 needs cuda 12.6 or lower

WORKDIR=${SLURM_SUBMIT_DIR:-.}
cd "${WORKDIR}"
echo "Running from this directory: ${WORKDIR}"
echo "Name of job: ${SLURM_JOB_NAME:-dynunet-brats}"
echo "ID of job: ${SLURM_JOB_ID:-local}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST:-local-node}"

module purge

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


#make sure uv is installed and available in the PATH. If not, install it using the following command:
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
srun --ntasks="${GPU_COUNT}" --ntasks-per-node="${GPU_COUNT}" --gpus-per-task=1 uv run python -m src.models.dynnet --batch-size 1 --num-workers 4 --num-samples 4 --roi-size 128 128 128 --wandb-mode online --run-name dynunet-brats-${SLURM_JOB_ID:-local} --data-dir /cluster/home/eldarja/CancerVision/res/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData

