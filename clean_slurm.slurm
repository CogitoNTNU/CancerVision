#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 3D"
#SBATCH --time=02:15:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --output=srun.out
#SBATCH --error=srun.err
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


#make sure uv is installed on your cluster
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run main.py --batch-size 32 --max-epochs 100 --num-workers 8
