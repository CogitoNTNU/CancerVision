#!/bin/bash
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

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
PROJECT_DIR="${WORKDIR}"

if [ ! -f "${PROJECT_DIR}/main.py" ] && [ -f "${WORKDIR}/CancerVision/main.py" ]; then
  PROJECT_DIR="${WORKDIR}/CancerVision"
fi

cd "${PROJECT_DIR}"
echo "Running from: ${PROJECT_DIR}"
echo "Job name: ${SLURM_JOB_NAME:-local}"
echo "Job id: ${SLURM_JOB_ID:-local}"
echo "Nodes: ${SLURM_JOB_NODELIST:-local}"

module purge
module load CUDA/12.4.0
module load cuDNN/9.1.0-CUDA-12.4.0

export PATH="$HOME/.local/bin:$PATH"
uv --version

uv sync

# Add uv-managed NVIDIA libs to linker path
SITE_PACKAGES="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
for d in \
  "$SITE_PACKAGES/nvidia/cudnn/lib" \
  "$SITE_PACKAGES/nvidia/cublas/lib" \
  "$SITE_PACKAGES/nvidia/cuda_runtime/lib" \
  "$SITE_PACKAGES/nvidia/cusparse/lib" \
  "$SITE_PACKAGES/nvidia/cufft/lib" \
  "$SITE_PACKAGES/nvidia/curand/lib" \
  "$SITE_PACKAGES/nvidia/nccl/lib"
do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

# Also add module-provided libs (if defined)
[ -n "${EBROOTCUDNN:-}" ] && export LD_LIBRARY_PATH="${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"
[ -n "${EBROOTCUDA:-}" ] && export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

# Hard fail early if libcudnn.so.9 still missing
if ! ldconfig -p 2>/dev/null | grep -q "libcudnn.so.9"; then
  if ! find "${PROJECT_DIR}/.venv" -name "libcudnn.so.9*" | grep -q .; then
    echo "ERROR: libcudnn.so.9 not found in linker cache or .venv."
    echo "Check torch CUDA deps / index selection in pyproject.toml."
    exit 1
  fi
fi

uv run python -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('libcudnn load ok')"

if [ -f "${PROJECT_DIR}/.env" ]; then
  set -a
  source <(sed 's/\r$//' "${PROJECT_DIR}/.env" | grep -v '^[[:space:]]*#' | grep -v '^[[:space:]]*$')
  set +a
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "================================================"
nvidia-smi
echo "================================================"
uv run python -c "import os, torch; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('wandb key present', bool(os.getenv('WANDB_API_KEY')))"
uv run python main.py --batch-size 32 --max-epochs 100 --num-workers 8
