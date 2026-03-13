#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="Cancer Vision UNet 2D"
#SBATCH --time=00:01:00
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

# Use current directory as fallback if not running via sbatch
SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-.}

# CONFIG
BRANCH="main"                                                                           # Default to `main` branch
PULL_ON_RUN=false 
SYNC_ON_RUN=false


usage() {
    echo "Usage: $0 [--branch=<name>] [--pull=true|false] [--sync=true|false]" >&2
    exit 1
}

# Parse flags
while [ "$#" -gt 0 ]; do
    case "$1" in
        --branch=*)
        BRANCH="${1#*=}"
        ;;
        --pull=*)
        PULL_ON_RUN="${1#*=}"
        ;;
        --sync=*)
        SYNC_ON_RUN="${1#*=}"
        ;;
        --help|-h)
        usage
        ;;
        *)
        echo "Unknown argument: $1" >&2
        usage
        ;;
    esac
    shift
done

# Validate booleans
case "$PULL_ON_RUN" in
    true|false) ;;
    *) echo "Error: --pull must be true or false" >&2; exit 1 ;;
esac

case "$SYNC_ON_RUN" in
    true|false) ;;
    *) echo "Error: --sync must be true or false" >&2; exit 1 ;;
esac


echo "CONFIGURATION:"
echo "BRANCH=$BRANCH"
echo "PULL_ON_RUN=$PULL_ON_RUN"
echo "SYNC_ON_RUN=$SYNC_ON_RUN"


ENV_PATH="/cluster/home/$(whoami)/CancerVision"


# Change to project directory
cd "CancerVision"


# Switch to the correct branch if not already on it
currentBranch=$(git rev-parse --abbrev-ref HEAD)
if [ "$currentBranch" != "$BRANCH" ]; then
    echo "Switching to branch $BRANCH..."
    git checkout $BRANCH
else 
    echo "Already on branch $BRANCH"
fi


# Pull latest changes if PULL_ON_RUN is true
if [ "$PULL_ON_RUN" = true ]; then
    echo "Fetching latest changes from GitHub..."
    git fetch origin $BRANCH
    git reset --hard origin/$BRANCH

    # Elevate perms of new run file
    chmod +x run_slurm.slurm
fi



# Load required modules
module purge

nvidia-smi

uv version

uv sync


# Check GPU info
echo "================================================"
echo "GPU Information:"
echo "================================================"
nvidia-smi
echo ""
echo "PyTorch CUDA availability:"

# python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "================================================"
echo ""

# Run the training script
uv run main.py

# ? # Deactivate the environment
# ? conda deactivate

echo "Job finished"
