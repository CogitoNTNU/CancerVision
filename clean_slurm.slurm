#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 2D"
#SBATCH --time=00:15:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=5G
#SBATCH --nodes=1
#SBATCH --output=test-srun.out
#SBATCH --error=test-srun.err
#SBATCH --mail-user=eldarja@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME" 
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

# Use current directory as fallback if not running via sbatch
SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-.}

# Load required modules
module purge

#login to wandb
#your wandb key in .env should be in the format: WANDB__API_KEY=your_key_here
set -a; source .env; set +a
uv sync
# Login to wandb
uv run wandb login --key "$WANDB_API_KEY" --relogin

# Run the training script
uv run main.py

# Check if the output files exist and print their contents
echo "Job finished"