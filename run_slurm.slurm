#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 2D"
#SBATCH --time=00:00:05
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
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
BRANCH="2d-segmentation"
PULL_ON_RUN=true
SYNC_ON_RUN=false


ENV_PATH="/cluster/work/$(whoami)/CancerVision"

# ? We need to have cloned the repository at least once to be able to run
# ? this file, so this doesn't make sense to have in the slurm script. 
# // Create repository if it doesn't exist
# // if [ ! -d "${ENV_PATH}" ]; then
# //     echo "Cloning project from GitHub..."
# //     git clone -b $BRANCH https://github.com/CogitoNTNU/CancerVision.git
# //     CLONED=true
# // fi

# // Change to project directory
# // cd "CancerVision"

# Switch to the correct branch if not already on it
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "$BRANCH" ]; then
    echo "Switching to branch $BRANCH..."
    git checkout $BRANCH
else 
    echo "Already on branch $BRANCH"
fi

# Pull latest changes if PULL_ON_RUN is true and CLONED is not true (to avoid pulling immediately after cloning)
if [ "$PULL_ON_RUN" = true ] && [ "${CLONED:-false}" != true ]; then
    echo "Pulling latest changes from GitHub..."
    git pull origin $BRANCH
fi

# # Load required modules
# module purge
# module load Anaconda3/2024.02-1

# # Create conda environment only if it doesn't exist
# if [ ! -d "${ENV_PATH}" ]; then
#     echo "Creating new conda environment..."
#     conda create -y --prefix ${ENV_PATH} python=3.11
# else
#     echo "Conda environment already exists, skipping creation."
# fi

# # Activate the environment
# source activate ${ENV_PATH}

# # Load environment variables from .env file (ignore comments and empty lines)
# if [ -f "${WORKDIR}/.env" ]; then
#     echo "Loading environment variables from .env..."
#     set -a  # Automatically export all variables
#     source <(grep -v '^#' ${WORKDIR}/.env | grep -v '^$')
#     set +a
# else
#     echo "Warning: .env file not found. W&B logging may not work."
# fi

# # Add project root to Python path so 'src' can be found as a module
# export PYTHONPATH="${WORKDIR}:${PYTHONPATH}"
# echo "PYTHONPATH set to: ${PYTHONPATH}"

# # Enable headless rendering for MuJoCo (EGL backend)
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl

# # Install/update dependencies
# echo "Installing/updating dependencies..."
# pip install -r requirements.txt

# echo "Installed packages:"
# pip freeze

# # Check GPU info
# echo "================================================"
# echo "GPU Information:"
# echo "================================================"
# nvidia-smi
# echo ""
# echo "PyTorch CUDA availability:"
# python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
# echo "================================================"
# echo ""

# # Run the training script
# python main.py

# # Deactivate the environment
# conda deactivate

# echo "Job finished"
