#!/bin/bash
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 3D"
#SBATCH --time=01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
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


# CONFIG
BRANCH="3d-segmentation"
PULL_ON_RUN=true
SYNC_ON_RUN=false

ENV_PATH="/cluster/home/$(whoami)/CancerVision"


# TODO: Add wrapper script to chmod +x and run this script with a single command, e.g. `./run_slurm.sh`

# Change to project directory
cd "CancerVision"


# #Switch to the correct branch if not already on it
# currentBranch=$(git rev-parse --abbrev-ref HEAD)
# if [ "$currentBranch" != "$BRANCH" ]; then
#     echo "Switching to branch $BRANCH..."
#     git checkout $BRANCH
# else 
#     echo "Already on branch $BRANCH"
# fi


# Pull latest changes if PULL_ON_RUN is true
if [ "$PULL_ON_RUN" = true ]; then
    echo "Fetching latest changes from GitHub..."
    git fetch origin $BRANCH
    git reset --hard origin/$BRANCH
    # Elevate perms of new run file
    chmod +x run_slurm.slurm
fi



# Load environment variables from .env file (ignore comments and empty lines)
if [ -f "${WORKDIR}/.env" ]; then
    echo "Loading environment variables from .env..."
    set -a  # Automatically export all variables
    source <(grep -v '^#' ${WORKDIR}/.env | grep -v '^$')
    set +a
else
    echo "Warning: .env file not found. W&B logging may not work."
fi

# Add project root to Python path so 'src' can be found as a module
echo "PYTHONPATH set to: ${PYTHONPATH}"

echo "Installed packages:"
pip freeze

# Check GPU info
echo "================================================"
echo "GPU Information:"
echo "================================================"
nvidia-smi
echo ""
echo "PyTorch CUDA availability:"
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "================================================"
echo ""

# Run the training script
uv run main.py

# Deactivate the environment
conda deactivate

echo "Job finished"
