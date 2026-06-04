"""CLI for metadata standardization and task-manifest generation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from .constants import (
    CFB_GBM_DEFAULT_ROOT,
    CFB_GBM_SOURCE_MANIFEST_DEFAULT,
    CFB_GBM_STANDARDIZED_DEFAULT_ROOT,
    BRATS2020_DEFAULT_ROOT,
    BRATS2020_SOURCE_MANIFEST_DEFAULT,
    BRATS2020_STANDARDIZED_DEFAULT_ROOT,
    BRATS2023_DEFAULT_ROOT,
    BRATS2023_SOURCE_MANIFEST_DEFAULT,
    BRATS2023_STANDARDIZED_DEFAULT_ROOT,
    BRATS2024_DEFAULT_ROOT,
    BRATS2024_SOURCE_MANIFEST_DEFAULT,
    BRATS2024_STANDARDIZED_DEFAULT_ROOT,
    BRAIN_STRUCTURE_DEFAULT_ROOT,
    BRAIN_STRUCTURE_SOURCE_MANIFEST_DEFAULT,
    BRAIN_STRUCTURE_STANDARDIZED_DEFAULT_ROOT,
    REMIND_DEFAULT_ROOT,
    REMIND_MASKS_DEFAULT_ROOT,
    REMIND_SOURCE_MANIFEST_DEFAULT,
    REMIND_STANDARDIZED_DEFAULT_ROOT,
    STANDARDIZED_DATASET_DEFAULT_ROOT,
    STANDARDIZED_TASK_MANIFESTS_DEFAULT_ROOT,
    UCSD_PTGBM_DEFAULT_ROOT,
    UCSD_PTGBM_SOURCE_MANIFEST_DEFAULT,
    UCSD_PTGBM_STANDARDIZED_DEFAULT_ROOT,
    UCSF_PDGM_DEFAULT_ROOT,
    UCSF_PDGM_SOURCE_MANIFEST_DEFAULT,
    UCSF_PDGM_STANDARDIZED_DEFAULT_ROOT,
    UTSW_GLIOMA_DEFAULT_ROOT,
    UTSW_GLIOMA_SOURCE_MANIFEST_DEFAULT,
    UTSW_GLIOMA_STANDARDIZED_DEFAULT_ROOT,
    UPENN_GBM_DEFAULT_ROOT,
    UPENN_GBM_SOURCE_MANIFEST_DEFAULT,
    UPENN_GBM_STANDARDIZED_DEFAULT_ROOT,
    VESTIBULAR_SCHWANNOMA_MC_RC2_DEFAULT_ROOT,
    VESTIBULAR_SCHWANNOMA_MC_RC2_SOURCE_MANIFEST_DEFAULT,
    VESTIBULAR_SCHWANNOMA_MC_RC2_STANDARDIZED_DEFAULT_ROOT,
    YALE_BRAIN_METS_LONGITUDINAL_DEFAULT_ROOT,
    YALE_BRAIN_METS_LONGITUDINAL_SOURCE_MANIFEST_DEFAULT,
    YALE_BRAIN_METS_LONGITUDINAL_STANDARDIZED_DEFAULT_ROOT,
)
from .io import read_csv_rows
from .pathing import resolve_existing_path, resolve_target_path
from .preprocess import (
    materialize_classification_manifest,
    materialize_segmentation_manifest,
    preflight_synthstrip_requirements,
)
from .registry import get_brats2020_adapter
from .registry import get_brats2023_adapter
from .registry import get_brats2024_adapter
from .registry import get_brain_structure_adapter
from .registry import get_cfb_gbm_adapter
from .registry import get_remind_adapter
from .registry import get_ucsd_ptgbm_adapter
from .registry import get_ucsf_pdgm_adapter
from .registry import get_utsw_glioma_adapter
from .registry import get_upenn_gbm_adapter
from .registry import get_vestibular_schwannoma_mc_rc2_adapter
from .registry import get_yale_brain_mets_longitudinal_adapter
from .task_manifests import write_task_manifests


def _missing_manifest_message(path: str | Path, error: FileNotFoundError) -> str:
    return (
        f"{error}\n"
        f"Likely cause: source manifest for this dataset has not been built yet."
    )


def _log_manifest_summary(rows: list[dict[str, str]]) -> None:
    classification_rows = [
        row for row in rows if str(row.get("task_type") or "classification").strip() == "classification"
    ]
    excluded_rows = [
        row for row in classification_rows if str(row.get("exclude_reason") or "").strip()
    ]
    dataset_counts = Counter(
        str(row.get("dataset_key") or "<missing>").strip() or "<missing>"
        for row in classification_rows
    )
    print(
        f"Loaded {len(rows)} manifest rows; classification rows={len(classification_rows)}, already excluded={len(excluded_rows)}",
        flush=True,
    )
    for dataset_key, count in sorted(dataset_counts.items()):
        print(
            f"  dataset {dataset_key}: {count} classification rows",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build standardization CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build mixed-MRI manifests and task CSVs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    brain_structure = subparsers.add_parser(
        "build-brain-structure-manifest",
        help="Build standardized manifest from brain-structure metadata.csv",
    )
    brain_structure.add_argument(
        "--brain-structure-root",
        default=BRAIN_STRUCTURE_DEFAULT_ROOT,
        help="Dataset root. Windows paths like Z:\\dataset\\brain-structure accepted.",
    )
    brain_structure.add_argument(
        "--metadata-name",
        default="metadata.csv",
        help="Metadata CSV filename relative to dataset root.",
    )
    brain_structure.add_argument(
        "--output-csv",
        default=BRAIN_STRUCTURE_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    brain_structure.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    brain_structure.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-brain validation.",
    )

    brats2020 = subparsers.add_parser(
        "build-brats2020-manifest",
        help="Build standardized manifest by crawling BraTS 2020 patient folders.",
    )
    brats2020.add_argument(
        "--brats2020-root",
        default=BRATS2020_DEFAULT_ROOT,
        help="BraTS 2020 root. Windows paths like Z:\\dataset\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData accepted.",
    )
    brats2020.add_argument(
        "--output-csv",
        default=BRATS2020_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    brats2020.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    brats2020.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    brats2023 = subparsers.add_parser(
        "build-brats2023-manifest",
        help="Build standardized manifest by crawling BraTS 2023 patient folders.",
    )
    brats2023.add_argument(
        "--brats2023-root",
        default=BRATS2023_DEFAULT_ROOT,
        help="BraTS 2023 root. Windows paths like Z:\\dataset\\brats2023\\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData accepted.",
    )
    brats2023.add_argument(
        "--output-csv",
        default=BRATS2023_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    brats2023.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    brats2023.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    brats2024 = subparsers.add_parser(
        "build-brats2024-manifest",
        help="Build standardized manifest by crawling BraTS 2024 patient folders.",
    )
    brats2024.add_argument(
        "--brats2024-root",
        default=BRATS2024_DEFAULT_ROOT,
        help="BraTS 2024 root. Windows paths like Z:\\dataset\\brats2024\\BraTS2024_small_dataset accepted.",
    )
    brats2024.add_argument(
        "--output-csv",
        default=BRATS2024_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    brats2024.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    brats2024.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    cfb_gbm = subparsers.add_parser(
        "build-cfb-gbm-manifest",
        help="Build standardized manifest by crawling CFB-GBM subject/visit folders.",
    )
    cfb_gbm.add_argument(
        "--cfb-gbm-root",
        default=CFB_GBM_DEFAULT_ROOT,
        help="Dataset root. Windows paths like Z:\\dataset\\PKG - CFB-GBM version 1\\CFB-GBM accepted.",
    )
    cfb_gbm.add_argument(
        "--output-csv",
        default=CFB_GBM_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    cfb_gbm.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    cfb_gbm.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    upenn_gbm = subparsers.add_parser(
        "build-upenn-gbm-manifest",
        help="Build standardized manifest by crawling PKG - UPENN-GBM-NIfTI.",
    )
    upenn_gbm.add_argument(
        "--upenn-gbm-root",
        default=UPENN_GBM_DEFAULT_ROOT,
        help="Dataset root. Windows paths like Z:\\dataset\\PKG - UPENN-GBM-NIfTI accepted.",
    )
    upenn_gbm.add_argument(
        "--output-csv",
        default=UPENN_GBM_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    upenn_gbm.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    upenn_gbm.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    ucsf_pdgm = subparsers.add_parser(
        "build-ucsf-pdgm-manifest",
        help="Build standardized manifest by crawling UCSF-PDGM-v5.",
    )
    ucsf_pdgm.add_argument(
        "--ucsf-pdgm-root",
        default=UCSF_PDGM_DEFAULT_ROOT,
        help="Dataset root. Windows paths like Z:\\dataset\\UCSF-PDGM-v5 accepted.",
    )
    ucsf_pdgm.add_argument(
        "--output-csv",
        default=UCSF_PDGM_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    ucsf_pdgm.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    ucsf_pdgm.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    ucsd_ptgbm = subparsers.add_parser(
        "build-ucsd-ptgbm-manifest",
        help="Build standardized manifest by crawling UCSD-PTGBM case folders.",
    )
    ucsd_ptgbm.add_argument(
        "--ucsd-ptgbm-root",
        default=UCSD_PTGBM_DEFAULT_ROOT,
        help="Dataset root. Windows paths like Z:\\dataset\\UCSD-PTGBM accepted.",
    )
    ucsd_ptgbm.add_argument(
        "--output-csv",
        default=UCSD_PTGBM_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    ucsd_ptgbm.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    ucsd_ptgbm.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    utsw_glioma = subparsers.add_parser(
        "build-utsw-glioma-manifest",
        help="Build standardized manifest by crawling UTSW-Glioma case folders.",
    )
    utsw_glioma.add_argument(
        "--utsw-glioma-root",
        default=UTSW_GLIOMA_DEFAULT_ROOT,
        help="Dataset root. Windows paths like Z:\\dataset\\UTSW-Glioma accepted.",
    )
    utsw_glioma.add_argument(
        "--output-csv",
        default=UTSW_GLIOMA_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    utsw_glioma.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    utsw_glioma.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    remind = subparsers.add_parser(
        "build-remind-manifest",
        help="Build standardized manifest by crawling ReMIND DICOM images and NRRD masks.",
    )
    remind.add_argument(
        "--remind-root",
        default=REMIND_DEFAULT_ROOT,
        help="ReMIND DICOM image root. Windows paths like Z:\\dataset\\remind accepted.",
    )
    remind.add_argument(
        "--remind-mask-root",
        default=REMIND_MASKS_DEFAULT_ROOT,
        help="ReMIND NRRD mask root. Defaults to extracted PKG - ReMIND_NRRD_Seg_Sep_2023 folder.",
    )
    remind.add_argument(
        "--output-csv",
        default=REMIND_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    remind.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )

    yale = subparsers.add_parser(
        "build-yale-brain-mets-longitudinal-manifest",
        help="Build standardized manifest by crawling Yale Brain Mets Longitudinal NIfTI package.",
    )
    yale.add_argument(
        "--yale-root",
        default=YALE_BRAIN_METS_LONGITUDINAL_DEFAULT_ROOT,
        help="Yale Brain Mets root. Windows paths like Z:\\dataset\\PKG - Yale-Brain-Mets-Longitudinal\\Yale-Brain-Mets-Longitudinal accepted.",
    )
    yale.add_argument(
        "--output-csv",
        default=YALE_BRAIN_METS_LONGITUDINAL_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    yale.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    yale.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    vestibular = subparsers.add_parser(
        "build-vestibular-schwannoma-mc-rc2-manifest",
        help="Build standardized manifest by crawling Vestibular-Schwannoma-MC-RC2 NIfTI package.",
    )
    vestibular.add_argument(
        "--vestibular-root",
        default=VESTIBULAR_SCHWANNOMA_MC_RC2_DEFAULT_ROOT,
        help="Vestibular Schwannoma root. Windows paths like Z:\\dataset\\Vestibular-Schwannoma-MC-RC2_Oct2025 accepted.",
    )
    vestibular.add_argument(
        "--output-csv",
        default=VESTIBULAR_SCHWANNOMA_MC_RC2_SOURCE_MANIFEST_DEFAULT,
        help="Destination CSV for standardized rows.",
    )
    vestibular.add_argument(
        "--include-excluded",
        action="store_true",
        help="Keep excluded rows in output with exclude_reason populated.",
    )
    vestibular.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip NIfTI nonzero-volume validation.",
    )

    preprocess_brain_structure = subparsers.add_parser(
        "preprocess-brain-structure-cls",
        help="Materialize cls/t1_128.nii.gz outputs for classification rows in manifest.",
    )
    preprocess_brain_structure.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing brain-structure rows.",
    )
    preprocess_brain_structure.add_argument(
        "--output-dir",
        default=BRAIN_STRUCTURE_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_brain_structure.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_brain_structure.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_brain_structure.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_brats2020 = subparsers.add_parser(
        "preprocess-brats2020-cls",
        help="Materialize cls/t1_128.nii.gz outputs for BraTS 2020 classification rows in manifest.",
    )
    preprocess_brats2020.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing BraTS 2020 rows.",
    )
    preprocess_brats2020.add_argument(
        "--output-dir",
        default=BRATS2020_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_brats2020.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_brats2020.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_brats2020.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_brats2023 = subparsers.add_parser(
        "preprocess-brats2023-cls",
        help="Materialize cls/t1_128.nii.gz outputs for BraTS 2023 classification rows in manifest.",
    )
    preprocess_brats2023.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing BraTS 2023 rows.",
    )
    preprocess_brats2023.add_argument(
        "--output-dir",
        default=BRATS2023_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_brats2023.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_brats2023.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_brats2023.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_brats2024 = subparsers.add_parser(
        "preprocess-brats2024-cls",
        help="Materialize cls/t1_128.nii.gz outputs for BraTS 2024 classification rows in manifest.",
    )
    preprocess_brats2024.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing BraTS 2024 rows.",
    )
    preprocess_brats2024.add_argument(
        "--output-dir",
        default=BRATS2024_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_brats2024.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_brats2024.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_brats2024.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_cfb_gbm = subparsers.add_parser(
        "preprocess-cfb-gbm-cls",
        help="Materialize cls/t1_128.nii.gz outputs for CFB-GBM classification rows in manifest.",
    )
    preprocess_cfb_gbm.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing CFB-GBM rows.",
    )
    preprocess_cfb_gbm.add_argument(
        "--output-dir",
        default=CFB_GBM_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_cfb_gbm.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_cfb_gbm.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_cfb_gbm.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_upenn_gbm = subparsers.add_parser(
        "preprocess-upenn-gbm-cls",
        help="Materialize cls/t1_128.nii.gz outputs for UPENN-GBM classification rows in manifest.",
    )
    preprocess_upenn_gbm.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing UPENN-GBM rows.",
    )
    preprocess_upenn_gbm.add_argument(
        "--output-dir",
        default=UPENN_GBM_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_upenn_gbm.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_upenn_gbm.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_upenn_gbm.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_ucsf_pdgm = subparsers.add_parser(
        "preprocess-ucsf-pdgm-cls",
        help="Materialize cls/t1_128.nii.gz outputs for UCSF-PDGM classification rows in manifest.",
    )
    preprocess_ucsf_pdgm.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing UCSF-PDGM rows.",
    )
    preprocess_ucsf_pdgm.add_argument(
        "--output-dir",
        default=UCSF_PDGM_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_ucsf_pdgm.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_ucsf_pdgm.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_ucsf_pdgm.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_ucsd_ptgbm = subparsers.add_parser(
        "preprocess-ucsd-ptgbm-cls",
        help="Materialize cls/t1_128.nii.gz outputs for UCSD-PTGBM classification rows in manifest.",
    )
    preprocess_ucsd_ptgbm.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing UCSD-PTGBM rows.",
    )
    preprocess_ucsd_ptgbm.add_argument(
        "--output-dir",
        default=UCSD_PTGBM_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_ucsd_ptgbm.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_ucsd_ptgbm.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_ucsd_ptgbm.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_utsw_glioma = subparsers.add_parser(
        "preprocess-utsw-glioma-cls",
        help="Materialize cls/t1_128.nii.gz outputs for UTSW-Glioma classification rows in manifest.",
    )
    preprocess_utsw_glioma.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing UTSW-Glioma rows.",
    )
    preprocess_utsw_glioma.add_argument(
        "--output-dir",
        default=UTSW_GLIOMA_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_utsw_glioma.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_utsw_glioma.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_utsw_glioma.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_remind = subparsers.add_parser(
        "preprocess-remind-cls",
        help="Materialize cls/t1_128.nii.gz outputs for ReMIND classification rows in manifest.",
    )
    preprocess_remind.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing ReMIND rows.",
    )
    preprocess_remind.add_argument(
        "--output-dir",
        default=REMIND_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_remind.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_remind.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_remind.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_yale = subparsers.add_parser(
        "preprocess-yale-brain-mets-longitudinal-cls",
        help="Materialize cls/t1_128.nii.gz outputs for Yale Brain Mets classification rows in manifest.",
    )
    preprocess_yale.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing Yale Brain Mets rows.",
    )
    preprocess_yale.add_argument(
        "--output-dir",
        default=YALE_BRAIN_METS_LONGITUDINAL_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_yale.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_yale.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_yale.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    preprocess_vestibular = subparsers.add_parser(
        "preprocess-vestibular-schwannoma-mc-rc2-cls",
        help="Materialize cls/t1_128.nii.gz outputs for Vestibular Schwannoma classification rows in manifest.",
    )
    preprocess_vestibular.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing Vestibular Schwannoma rows.",
    )
    preprocess_vestibular.add_argument(
        "--output-dir",
        default=VESTIBULAR_SCHWANNOMA_MC_RC2_STANDARDIZED_DEFAULT_ROOT,
        help="Root directory where <global_case_id>/cls outputs are written.",
    )
    preprocess_vestibular.add_argument(
        "--skip-mask",
        action="store_true",
        help="Do not write cls/brain_mask_128.nii.gz.",
    )
    preprocess_vestibular.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized cls outputs. Defaults inside output dir.",
    )
    preprocess_vestibular.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for skull-on classification datasets.",
    )

    task_manifests = subparsers.add_parser(
        "build-task-manifests",
        help="Build downstream task manifests from standardized manifest rows.",
    )
    task_manifests.add_argument(
        "--input-manifest",
        required=True,
        help="CSV with standardized rows from one or more datasets.",
    )
    task_manifests.add_argument(
        "--output-dir",
        default=STANDARDIZED_TASK_MANIFESTS_DEFAULT_ROOT,
        help="Directory where task CSVs are written.",
    )
    task_manifests.add_argument(
        "--include-any-unhealthy",
        action="store_true",
        help="Also write optional classification_t1_any_unhealthy_vs_healthy.csv.",
    )

    preprocess_segmentation_native = subparsers.add_parser(
        "materialize-segmentation-native",
        help="Convert native-scale segmentation image/mask pairs into `.nii.gz` standardized layout.",
    )
    preprocess_segmentation_native.add_argument(
        "--input-manifest",
        required=True,
        help="Standardized manifest CSV containing segmentation rows.",
    )
    preprocess_segmentation_native.add_argument(
        "--output-dir",
        default=rf"{STANDARDIZED_DATASET_DEFAULT_ROOT}\segmentation_native",
        help="Root directory where <global_case_id>/seg outputs are written.",
    )
    preprocess_segmentation_native.add_argument(
        "--output-manifest",
        default=None,
        help="Optional manifest path pointing at materialized seg outputs. Defaults inside output dir.",
    )
    preprocess_segmentation_native.add_argument(
        "--synthstrip-cmd",
        default="mri_synthstrip",
        help="External SynthStrip command to use for segmentation datasets that require skull stripping.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for standardization CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-brain-structure-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=BRAIN_STRUCTURE_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_brain_structure_adapter(
            args.brain_structure_root,
            metadata_filename=args.metadata_name,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} brain-structure rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-brats2020-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=BRATS2020_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_brats2020_adapter(args.brats2020_root)
        print(
            f"Scanning BraTS 2020 root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} BraTS 2020 rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-brats2023-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=BRATS2023_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_brats2023_adapter(args.brats2023_root)
        print(
            f"Scanning BraTS 2023 root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} BraTS 2023 rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-brats2024-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=BRATS2024_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_brats2024_adapter(args.brats2024_root)
        print(
            f"Scanning BraTS 2024 root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} BraTS 2024 rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-cfb-gbm-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=CFB_GBM_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_cfb_gbm_adapter(args.cfb_gbm_root)
        print(
            f"Scanning CFB-GBM root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} CFB-GBM rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-upenn-gbm-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=UPENN_GBM_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_upenn_gbm_adapter(args.upenn_gbm_root)
        print(
            f"Scanning UPENN-GBM root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} UPENN-GBM rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-ucsf-pdgm-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=UCSF_PDGM_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_ucsf_pdgm_adapter(args.ucsf_pdgm_root)
        print(
            f"Scanning UCSF-PDGM root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} UCSF-PDGM rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-ucsd-ptgbm-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=UCSD_PTGBM_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_ucsd_ptgbm_adapter(args.ucsd_ptgbm_root)
        print(
            f"Scanning UCSD-PTGBM root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} UCSD-PTGBM rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-utsw-glioma-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=UTSW_GLIOMA_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_utsw_glioma_adapter(args.utsw_glioma_root)
        print(
            f"Scanning UTSW-Glioma root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} UTSW-Glioma rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-remind-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=REMIND_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_remind_adapter(
            args.remind_root,
            mask_root=args.remind_mask_root,
        )
        print(
            f"Scanning ReMIND images: {adapter.image_root}",
            flush=True,
        )
        print(
            f"Scanning ReMIND masks: {adapter.mask_root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
        )
        print(
            f"Wrote {len(records)} ReMIND rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-yale-brain-mets-longitudinal-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=YALE_BRAIN_METS_LONGITUDINAL_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_yale_brain_mets_longitudinal_adapter(args.yale_root)
        print(
            f"Scanning Yale Brain Mets root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} Yale Brain Mets rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-vestibular-schwannoma-mc-rc2-manifest":
        output_csv = resolve_target_path(
            args.output_csv,
            default=VESTIBULAR_SCHWANNOMA_MC_RC2_SOURCE_MANIFEST_DEFAULT,
        )
        adapter = get_vestibular_schwannoma_mc_rc2_adapter(args.vestibular_root)
        print(
            f"Scanning Vestibular Schwannoma root: {adapter.root}",
            flush=True,
        )
        records = adapter.write_manifest(
            output_csv,
            include_excluded=args.include_excluded,
            validate_images=not args.skip_image_validation,
        )
        print(
            f"Wrote {len(records)} Vestibular Schwannoma rows to {output_csv}",
            flush=True,
        )
        return

    if args.command == "build-task-manifests":
        try:
            input_manifest = resolve_existing_path(args.input_manifest)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                _missing_manifest_message(args.input_manifest, exc)
            ) from exc
        output_dir = resolve_target_path(
            args.output_dir,
            default=STANDARDIZED_TASK_MANIFESTS_DEFAULT_ROOT,
        )
        rows = read_csv_rows(input_manifest)
        manifests = write_task_manifests(
            rows,
            output_dir,
            include_any_unhealthy=args.include_any_unhealthy,
        )
        print(
            f"Wrote {len(manifests)} task manifests to {output_dir}",
            flush=True,
        )
        return

    if args.command == "materialize-segmentation-native":
        try:
            input_manifest = resolve_existing_path(args.input_manifest)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                _missing_manifest_message(args.input_manifest, exc)
            ) from exc
        output_dir = resolve_target_path(
            args.output_dir,
            default=rf"{STANDARDIZED_DATASET_DEFAULT_ROOT}\segmentation_native",
        )
        output_manifest = None
        if args.output_manifest:
            output_manifest = resolve_target_path(
                args.output_manifest,
                default=args.output_manifest,
            )
        print(
            f"Reading manifest: {input_manifest}",
            flush=True,
        )
        print(
            f"Writing native segmentation dataset to: {output_dir}",
            flush=True,
        )
        rows = read_csv_rows(input_manifest)
        materialized_rows, manifest_path = materialize_segmentation_manifest(
            rows,
            output_dir,
            output_manifest_path=output_manifest,
            synthstrip_cmd=args.synthstrip_cmd,
        )
        print(
            f"Wrote {len(materialized_rows)} segmentation pairs and manifest to {manifest_path}",
            flush=True,
        )
        return

    if args.command in {
        "preprocess-brain-structure-cls",
        "preprocess-brats2020-cls",
        "preprocess-brats2023-cls",
        "preprocess-brats2024-cls",
        "preprocess-cfb-gbm-cls",
        "preprocess-upenn-gbm-cls",
        "preprocess-ucsf-pdgm-cls",
        "preprocess-ucsd-ptgbm-cls",
        "preprocess-utsw-glioma-cls",
        "preprocess-remind-cls",
        "preprocess-yale-brain-mets-longitudinal-cls",
        "preprocess-vestibular-schwannoma-mc-rc2-cls",
    }:
        try:
            input_manifest = resolve_existing_path(args.input_manifest)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                _missing_manifest_message(args.input_manifest, exc)
            ) from exc
        default_output_dir = (
            BRAIN_STRUCTURE_STANDARDIZED_DEFAULT_ROOT
            if args.command == "preprocess-brain-structure-cls"
            else (
                BRATS2020_STANDARDIZED_DEFAULT_ROOT
                if args.command == "preprocess-brats2020-cls"
                else (
                    BRATS2023_STANDARDIZED_DEFAULT_ROOT
                    if args.command == "preprocess-brats2023-cls"
                    else (
                        BRATS2024_STANDARDIZED_DEFAULT_ROOT
                        if args.command == "preprocess-brats2024-cls"
                        else (
                            CFB_GBM_STANDARDIZED_DEFAULT_ROOT
                            if args.command == "preprocess-cfb-gbm-cls"
                            else (
                                UPENN_GBM_STANDARDIZED_DEFAULT_ROOT
                                if args.command == "preprocess-upenn-gbm-cls"
                                else (
                                    UCSF_PDGM_STANDARDIZED_DEFAULT_ROOT
                                    if args.command == "preprocess-ucsf-pdgm-cls"
                                    else (
                                        UCSD_PTGBM_STANDARDIZED_DEFAULT_ROOT
                                        if args.command == "preprocess-ucsd-ptgbm-cls"
                                        else (
                                            UTSW_GLIOMA_STANDARDIZED_DEFAULT_ROOT
                                            if args.command == "preprocess-utsw-glioma-cls"
                                            else (
                                                REMIND_STANDARDIZED_DEFAULT_ROOT
                                                if args.command == "preprocess-remind-cls"
                                                else (
                                                    YALE_BRAIN_METS_LONGITUDINAL_STANDARDIZED_DEFAULT_ROOT
                                                    if args.command
                                                    == "preprocess-yale-brain-mets-longitudinal-cls"
                                                    else VESTIBULAR_SCHWANNOMA_MC_RC2_STANDARDIZED_DEFAULT_ROOT
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        output_dir = resolve_target_path(
            args.output_dir,
            default=default_output_dir,
        )
        output_manifest = None
        if args.output_manifest:
            output_manifest = resolve_target_path(
                args.output_manifest,
                default=args.output_manifest,
            )
        print(
            f"Reading manifest: {input_manifest}",
            flush=True,
        )
        print(
            f"Writing standardized dataset to: {output_dir}",
            flush=True,
        )
        rows = read_csv_rows(input_manifest)
        _log_manifest_summary(rows)
        preflight_synthstrip_requirements(
            rows,
            synthstrip_cmd=args.synthstrip_cmd,
        )
        materialized_rows, manifest_path = materialize_classification_manifest(
            rows,
            output_dir,
            save_mask=not args.skip_mask,
            output_manifest_path=output_manifest,
            synthstrip_cmd=args.synthstrip_cmd,
        )
        print(
            f"Wrote {len(materialized_rows)} classification views and manifest to {manifest_path}",
            flush=True,
        )
        return

    raise RuntimeError(f"Unhandled command: {args.command}")
