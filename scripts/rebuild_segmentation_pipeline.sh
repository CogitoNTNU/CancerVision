#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKDIR}"

STANDARDIZED_ROOT="${STANDARDIZED_ROOT:-${WORKDIR}/res/dataset/cancervision-standardized}"
MANIFEST_DIR="${MANIFEST_DIR:-${STANDARDIZED_ROOT}/manifests}"
TASK_DIR="${TASK_DIR:-${STANDARDIZED_ROOT}/task_manifests}"
SEG_NATIVE_DIR="${SEG_NATIVE_DIR:-${STANDARDIZED_ROOT}/segmentation_native}"

MERGED_SOURCE_MANIFEST="${MERGED_SOURCE_MANIFEST:-${MANIFEST_DIR}/all_sources_merged.csv}"
SEG_TASK_MANIFEST="${SEG_TASK_MANIFEST:-${TASK_DIR}/segmentation_binary_curated.csv}"
SEG_MATERIALIZED_MANIFEST="${SEG_MATERIALIZED_MANIFEST:-${SEG_NATIVE_DIR}/segmentation_materialized_manifest.csv}"

BRATS2020_ROOT="${BRATS2020_ROOT:-Z:\dataset\brats2020}"
BRATS2023_ROOT="${BRATS2023_ROOT:-Z:\dataset\brats2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData}"
BRATS2024_ROOT="${BRATS2024_ROOT:-Z:\dataset\brats2024\BraTS2024_small_dataset}"
CFB_GBM_ROOT="${CFB_GBM_ROOT:-Z:\dataset\PKG - CFB-GBM version 1\CFB-GBM}"
UPENN_GBM_ROOT="${UPENN_GBM_ROOT:-Z:\dataset\PKG - UPENN-GBM-NIfTI}"
UCSF_PDGM_ROOT="${UCSF_PDGM_ROOT:-Z:\dataset\UCSF-PDGM-v5}"
UCSD_PTGBM_ROOT="${UCSD_PTGBM_ROOT:-Z:\dataset\UCSD-PTGBM}"
UTSW_GLIOMA_ROOT="${UTSW_GLIOMA_ROOT:-Z:\dataset\UTSW-Glioma}"
REMIND_ROOT="${REMIND_ROOT:-Z:\dataset\remind}"
REMIND_MASK_ROOT="${REMIND_MASK_ROOT:-Z:\dataset\PKG - ReMIND_NRRD_Seg_Sep_2023\ReMIND_NRRD_Seg_Sep_2023}"
YALE_ROOT="${YALE_ROOT:-Z:\dataset\PKG - Yale-Brain-Mets-Longitudinal\Yale-Brain-Mets-Longitudinal}"
VESTIBULAR_ROOT="${VESTIBULAR_ROOT:-Z:\dataset\Vestibular-Schwannoma-MC-RC2_Oct2025}"

FORCE_REBUILD="${FORCE_REBUILD:-0}"

to_shell_path() {
  local raw_path="$1"

  if [[ -z "${raw_path}" ]]; then
    printf '%s' ""
    return
  fi

  if [[ "${raw_path}" =~ ^([A-Za-z]):[\\/](.*)$ ]]; then
    local drive_letter
    local tail_path
    drive_letter="${BASH_REMATCH[1],,}"
    tail_path="${BASH_REMATCH[2]//\\//}"
    printf '/mnt/%s/%s' "${drive_letter}" "${tail_path}"
    return
  fi

  if [[ "${raw_path}" =~ ^\\\\([^\\]+)\\([^\\]+)\\?(.*)$ ]]; then
    local server_name
    local share_name
    local tail_path
    server_name="${BASH_REMATCH[1]}"
    share_name="${BASH_REMATCH[2]}"
    tail_path="${BASH_REMATCH[3]//\\//}"
    if [[ -n "${tail_path}" ]]; then
      printf '//%s/%s/%s' "${server_name}" "${share_name}" "${tail_path}"
    else
      printf '//%s/%s' "${server_name}" "${share_name}"
    fi
    return
  fi

  printf '%s' "${raw_path}"
}

MANIFEST_DIR_SHELL="$(to_shell_path "${MANIFEST_DIR}")"
TASK_DIR_SHELL="$(to_shell_path "${TASK_DIR}")"
SEG_NATIVE_DIR_SHELL="$(to_shell_path "${SEG_NATIVE_DIR}")"
MERGED_SOURCE_MANIFEST_SHELL="$(to_shell_path "${MERGED_SOURCE_MANIFEST}")"
SEG_TASK_MANIFEST_SHELL="$(to_shell_path "${SEG_TASK_MANIFEST}")"
SEG_MATERIALIZED_MANIFEST_SHELL="$(to_shell_path "${SEG_MATERIALIZED_MANIFEST}")"

mkdir -p "${MANIFEST_DIR_SHELL}" "${TASK_DIR_SHELL}" "${SEG_NATIVE_DIR_SHELL}"

if command -v uv >/dev/null 2>&1; then
  PY_RUNNER=(uv run python)
elif command -v python >/dev/null 2>&1; then
  PY_RUNNER=(python)
elif command -v python3 >/dev/null 2>&1; then
  PY_RUNNER=(python3)
else
  echo "Missing python/uv in PATH"
  exit 1
fi

SYNTHSTRIP_CMD="${SYNTHSTRIP_CMD:-${PY_RUNNER[*]} scripts/synthstrip_docker.py}"

if [[ "${FORCE_REBUILD}" == "1" ]]; then
  echo "FORCE_REBUILD=1 -> removing old merged/task/materialized outputs"
  rm -f "${MERGED_SOURCE_MANIFEST_SHELL}" "${SEG_TASK_MANIFEST_SHELL}" "${SEG_MATERIALIZED_MANIFEST_SHELL}"
  rm -rf "${SEG_NATIVE_DIR_SHELL}"
  mkdir -p "${SEG_NATIVE_DIR_SHELL}"
fi

BUILT_MANIFESTS=()

run_build() {
  local name="$1"
  local root_path="$2"
  local output_csv="$3"
  shift 3

  if [[ "${FORCE_REBUILD}" != "1" && -f "$(to_shell_path "${output_csv}")" ]]; then
    echo "reuse ${name}: ${output_csv}"
    BUILT_MANIFESTS+=("${output_csv}")
    return
  fi
  if [[ -z "${root_path}" ]]; then
    echo "skip ${name}: root empty"
    return
  fi
  local root_shell
  root_shell="$(to_shell_path "${root_path}")"
  if [[ ! -e "${root_shell}" ]]; then
    echo "skip ${name}: root missing -> ${root_path}"
    return
  fi

  echo "build ${name}: ${root_path}"
  "${PY_RUNNER[@]}" main.py standardize "$@" --output-csv "${output_csv}" --include-excluded
  BUILT_MANIFESTS+=("${output_csv}")
}

run_build \
  "brats2020" \
  "${BRATS2020_ROOT}" \
  "${MANIFEST_DIR}/brats2020_source_manifest.csv" \
  build-brats2020-manifest \
  --brats2020-root "${BRATS2020_ROOT}"

run_build \
  "brats2023" \
  "${BRATS2023_ROOT}" \
  "${MANIFEST_DIR}/brats2023_source_manifest.csv" \
  build-brats2023-manifest \
  --brats2023-root "${BRATS2023_ROOT}"

run_build \
  "brats2024" \
  "${BRATS2024_ROOT}" \
  "${MANIFEST_DIR}/brats2024_source_manifest.csv" \
  build-brats2024-manifest \
  --brats2024-root "${BRATS2024_ROOT}"

run_build \
  "cfb_gbm" \
  "${CFB_GBM_ROOT}" \
  "${MANIFEST_DIR}/cfb_gbm_source_manifest.csv" \
  build-cfb-gbm-manifest \
  --cfb-gbm-root "${CFB_GBM_ROOT}"

run_build \
  "upenn_gbm" \
  "${UPENN_GBM_ROOT}" \
  "${MANIFEST_DIR}/upenn_gbm_source_manifest.csv" \
  build-upenn-gbm-manifest \
  --upenn-gbm-root "${UPENN_GBM_ROOT}"

run_build \
  "ucsf_pdgm" \
  "${UCSF_PDGM_ROOT}" \
  "${MANIFEST_DIR}/ucsf_pdgm_source_manifest.csv" \
  build-ucsf-pdgm-manifest \
  --ucsf-pdgm-root "${UCSF_PDGM_ROOT}"

run_build \
  "ucsd_ptgbm" \
  "${UCSD_PTGBM_ROOT}" \
  "${MANIFEST_DIR}/ucsd_ptgbm_source_manifest.csv" \
  build-ucsd-ptgbm-manifest \
  --ucsd-ptgbm-root "${UCSD_PTGBM_ROOT}"

run_build \
  "utsw_glioma" \
  "${UTSW_GLIOMA_ROOT}" \
  "${MANIFEST_DIR}/utsw_glioma_source_manifest.csv" \
  build-utsw-glioma-manifest \
  --utsw-glioma-root "${UTSW_GLIOMA_ROOT}"

run_build \
  "remind" \
  "${REMIND_ROOT}" \
  "${MANIFEST_DIR}/remind_source_manifest.csv" \
  build-remind-manifest \
  --remind-root "${REMIND_ROOT}" \
  --remind-mask-root "${REMIND_MASK_ROOT}"

run_build \
  "yale_brain_mets_longitudinal" \
  "${YALE_ROOT}" \
  "${MANIFEST_DIR}/yale_brain_mets_longitudinal_source_manifest.csv" \
  build-yale-brain-mets-longitudinal-manifest \
  --yale-root "${YALE_ROOT}"

run_build \
  "vestibular_schwannoma_mc_rc2" \
  "${VESTIBULAR_ROOT}" \
  "${MANIFEST_DIR}/vestibular_schwannoma_mc_rc2_source_manifest.csv" \
  build-vestibular-schwannoma-mc-rc2-manifest \
  --vestibular-root "${VESTIBULAR_ROOT}"

if [[ "${#BUILT_MANIFESTS[@]}" -eq 0 ]]; then
  echo "No source manifests built. Set dataset root env vars first."
  exit 1
fi

echo "merge source manifests -> ${MERGED_SOURCE_MANIFEST}"
"${PY_RUNNER[@]}" scripts/merge_standardized_manifests.py \
  --output "${MERGED_SOURCE_MANIFEST}" \
  "${BUILT_MANIFESTS[@]}"

echo "build seg task manifest -> ${TASK_DIR}"
"${PY_RUNNER[@]}" main.py standardize build-task-manifests \
  --input-manifest "${MERGED_SOURCE_MANIFEST}" \
  --output-dir "${TASK_DIR}"

echo "materialize seg images -> ${SEG_NATIVE_DIR}"
"${PY_RUNNER[@]}" main.py standardize materialize-segmentation-native \
  --input-manifest "${SEG_TASK_MANIFEST}" \
  --output-dir "${SEG_NATIVE_DIR}" \
  --output-manifest "${SEG_MATERIALIZED_MANIFEST}" \
  --synthstrip-cmd "${SYNTHSTRIP_CMD}"

echo
echo "done"
echo "merged source manifest: ${MERGED_SOURCE_MANIFEST}"
echo "seg task manifest: ${SEG_TASK_MANIFEST}"
echo "seg materialized manifest: ${SEG_MATERIALIZED_MANIFEST}"
