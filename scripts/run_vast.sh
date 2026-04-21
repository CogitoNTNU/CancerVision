#!/usr/bin/env bash
# Vast.ai single-command pipeline entrypoint.
#
# Usage from a fresh vast.ai instance:
#   git clone <repo> && cd cancervision
#   cp .env.example .env       # put WANDB_API_KEY / DATA_DIR / etc. here
#   bash scripts/run_vast.sh   # or: bash scripts/run_vast.sh --max-epochs 50
#
# All extra CLI args are forwarded to `python -m src.pipeline`, which in turn
# forwards them to `src.training.train`. Defaults assume a single CUDA GPU.
#
# Env vars (either exported or set in .env):
#   WANDB_API_KEY     required for --wandb-mode=online (omit for offline)
#   WANDB_ENTITY      optional (defaults to "cancervision")
#   DATA_DIR          BraTS root; default res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData
#   DATA_ARCHIVE      optional path or URL to a BraTS archive to extract on first run

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

if [ -f "${WORKDIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${WORKDIR}/.env"
  set +a
fi

: "${DATA_DIR:=${WORKDIR}/res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData}"
: "${WANDB_MODE:=online}"
export DATA_DIR WANDB_MODE

echo "== vast.ai pipeline =="
echo "Workdir   : ${WORKDIR}"
echo "Data dir  : ${DATA_DIR}"
echo "W&B mode  : ${WANDB_MODE}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "== syncing python environment =="
uv sync --frozen

if [ -n "${DATA_ARCHIVE:-}" ] && [ ! -d "${DATA_DIR}" ]; then
  echo "== fetching dataset archive =="
  mkdir -p "$(dirname "${DATA_DIR}")"
  archive_local="${WORKDIR}/res/data/_archive"
  if [[ "${DATA_ARCHIVE}" =~ ^https?:// ]]; then
    mkdir -p "${archive_local}"
    curl -L --fail -o "${archive_local}/brats.archive" "${DATA_ARCHIVE}"
    src_archive="${archive_local}/brats.archive"
  else
    src_archive="${DATA_ARCHIVE}"
  fi
  case "${src_archive}" in
    *.tar.gz|*.tgz) tar -xzf "${src_archive}" -C "$(dirname "${DATA_DIR}")" ;;
    *.tar)          tar -xf  "${src_archive}" -C "$(dirname "${DATA_DIR}")" ;;
    *.zip)          unzip -q "${src_archive}" -d "$(dirname "${DATA_DIR}")" ;;
    *) echo "Unsupported DATA_ARCHIVE format: ${src_archive}" >&2; exit 1 ;;
  esac
fi

echo "== GPU visibility =="
nvidia-smi || { echo "nvidia-smi failed; this entrypoint expects a CUDA GPU." >&2; exit 1; }

RUN_NAME="${RUN_NAME:-cancervision-$(date -u +%Y%m%d-%H%M%S)}"

echo "== launching training =="
exec uv run python -m src.pipeline \
  --data-dir "${DATA_DIR}" \
  --wandb-mode "${WANDB_MODE}" \
  --run-name "${RUN_NAME}" \
  "$@"
