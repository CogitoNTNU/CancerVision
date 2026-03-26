#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 3D"
#SBATCH --time=06:15:00
#SBATCH --partition=GPUQ
#SBATCH --job-name="dynunet-brats"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_terminal.out
#SBATCH --error=slurm_terminal.err
#SBATCH --mail-user=eldarja@ntnu.no
#SBATCH --mail-type=ALL




# p100 needs cuda 12.6 or lower

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME" 
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# Use current directory as fallback if not running via sbatch
SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-.}
module purge


#make sure uv is installed and available in the PATH. If not, install it using the following command:
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
srun uv run python -m src.models.dynnet --batch-size 1 --num-workers 4 --num-samples 4 --roi-size 128 128 128 --wandb-mode online --run-name dynunet-brats-${SLURM_JOB_ID} --data-dir /cluster/home/eldarja/CancerVision/res/dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData

