#!/bin/bash
#SBATCH --job-name=brats-unet
#SBATCH --output=logs/brats_%j.out
#SBATCH --error=logs/brats_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ──────────────────────────────────────────────────────────────
#  BraTS 2020 2D U-Net Training — SLURM batch script
#
#  Submit from the project root:
#      sbatch scripts/train_brats.slurm
#
#  Or via SSH:
#      ssh <user>@<host> "cd /path/to/cancervision && sbatch scripts/train_brats.slurm"
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

echo "============================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Job Name    : $SLURM_JOB_NAME"
echo "Node        : $(hostname)"
echo "GPUs        : $SLURM_GPUS_ON_NODE"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Project dir : $PROJECT_DIR"
echo "Date        : $(date)"
echo "============================================"

# ── Environment ──────────────────────────────────────────────
# Activate a virtual-env / conda env — uncomment & edit the one
# that matches your cluster setup.

# Option A: venv
# source .venv/bin/activate

# Option B: conda / mamba
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate cancervision

# Option C: module system
module load python/3.13 cuda/12.x
source .venv/bin/activate

# ── Create log directory ─────────────────────────────────────
mkdir -p logs

# ── Verify GPU visibility ───────────────────────────────────
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}  |  Devices: {torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Launch training ──────────────────────────────────────────
python src/models/train_brats.py \
    --data-dir  res/data/archive/BraTS2020_training_data/content/data \
    --save-dir  res/models \
    --max-epochs 10 \
    --batch-size 16 \
    --lr 1e-4 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --seed 0

echo ""
echo "Done — $(date)"
