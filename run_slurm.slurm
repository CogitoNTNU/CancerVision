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

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR}"
PROJECT_DIR="${WORKDIR}/CancerVision"

BRANCH="3d-segmentation"
PULL_ON_RUN=true

echo "Running from: ${WORKDIR}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Job id: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"

cd "${PROJECT_DIR}"

if [ "${PULL_ON_RUN}" = true ]; then
  git fetch origin "${BRANCH}"
  git reset --hard "origin/${BRANCH}"
  chmod +x run_slurm.slurm
fi

# Cluster GPU runtime
module purge
module load CUDA/12.4.0
module load cuDNN/9.1.0-CUDA-12.4.0

export PATH="$HOME/.local/bin:$PATH"
uv --version

# Reproducible env from lock
uv sync --frozen

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

echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Hard fail early if libcudnn.so.9 still missing
if ! ldconfig -p 2>/dev/null | grep -q "libcudnn.so.9"; then
  if ! find "${PROJECT_DIR}/.venv" -name "libcudnn.so.9*" | grep -q .; then
    echo "ERROR: libcudnn.so.9 not found in linker cache or .venv."
    echo "Check that torch CUDA deps are installed in uv.lock."
    exit 1
  fi
fi

if [ -f "${WORKDIR}/.env" ]; then
  set -a
  source <(grep -v '^#' "${WORKDIR}/.env" | grep -v '^$')
  set +a
fi

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

echo "================================================"
nvidia-smi
echo "================================================"
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}')"

uv run python main.py

echo "Job finished"