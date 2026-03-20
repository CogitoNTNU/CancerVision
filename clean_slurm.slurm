#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 3D"
#SBATCH --time=02:15:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
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


#make sure uv is installed on your cluster
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python -c "import torch, os; print('torch ok', torch.version); print('key?', bool(os.getenv('WANDB_API_KEY')))"
uv run main.py --batch-size 1 --num-workers 0 --num-samples 1 --roi-size 96 96 96
