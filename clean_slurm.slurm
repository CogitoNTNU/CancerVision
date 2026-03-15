#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 2D"
#SBATCH --time=00:01:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=5G
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_combined.txt
#SBATCH --error=slurm_outputs/output_combined.err
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
module load python/3.13.5

pip install uv

uv sync

#login to wandb
uv run wandb login --relogin

# Run the training script
uv run main.py

stdout_file="slurm_outputs/output_combined.txt"
stderr_file="slurm_outputs/output_combined.err"

# Check if the output files exist and print their contents
echo "Job finished"