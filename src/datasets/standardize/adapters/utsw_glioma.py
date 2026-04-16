"""Adapter for UTSW-Glioma case folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np

from ..constants import (
    UTSW_GLIOMA_DATASET_KEY,
    UTSW_GLIOMA_PREPROC_PROFILE,
    UTSW_GLIOMA_SEG_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

NIFTI_SUFFIXES = (".nii", ".nii.gz")
STRUCTURAL_ANCHOR_PRIORITY = ("t1ce", "t1", "t2", "flair")
MASK_PRIORITY = (
    "tumorseg_manual_correction",
    "rtumorseg_manual_correction",
    "tumorseg_fets",
)
SERIES_FILENAMES = {
    "brain_t1.nii.gz": "t1",
    "brain_t1ce.nii.gz": "t1ce",
    "brain_t2.nii.gz": "t2",
    "brain_flair.nii.gz": "flair",
    "tumorseg_manual_correction.nii.gz": "tumorseg_manual_correction",
    "rtumorseg_manual_correction.nii.gz": "rtumorseg_manual_correction",
    "tumorseg_FeTS.nii.gz": "tumorseg_fets",
}


def _has_nonzero_volume(image_path: Path) -> bool:
    data = np.asanyarray(nib.load(str(image_path)).dataobj)
    return bool(np.any(data != 0))


def _image_validation_exclude_reason(image_path: Path) -> str:
    try:
        if _has_nonzero_volume(image_path):
            return ""
        return "empty_brain_after_load"
    except (ImageFileError, OSError, EOFError, ValueError):
        return "invalid_image_file"


@dataclass(slots=True)
class UtswGliomaCase:
    subject_id: str
    files: dict[str, Path]

    @property
    def subject_key(self) -> str:
        return f"{UTSW_GLIOMA_DATASET_KEY}:{self.subject_id}"

    @property
    def global_case_id(self) -> str:
        return f"{UTSW_GLIOMA_DATASET_KEY}__{self.subject_id}__baseline"

    def get_t1_path(self) -> Path | None:
        return self.files.get("t1")

    def get_anchor_path(self) -> Path | None:
        for modality in STRUCTURAL_ANCHOR_PRIORITY:
            path = self.files.get(modality)
            if path is not None:
                return path
        return None

    def get_mask_key(self) -> str | None:
        for mask_key in MASK_PRIORITY:
            if mask_key in self.files:
                return mask_key
        return None

    def get_mask_path(self) -> Path | None:
        mask_key = self.get_mask_key()
        if mask_key is None:
            return None
        return self.files[mask_key]


class UtswGliomaAdapter:
    """Crawl UTSW-Glioma case folders and emit standardized rows."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover_cases(self) -> list[UtswGliomaCase]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"UTSW-Glioma root not found: {self.root}")

        cases: list[UtswGliomaCase] = []
        for case_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            files: dict[str, Path] = {}
            for path in sorted(case_dir.iterdir()):
                if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                    continue
                series_key = SERIES_FILENAMES.get(path.name)
                if series_key is None:
                    continue
                files.setdefault(series_key, path)
            if files:
                cases.append(UtswGliomaCase(subject_id=case_dir.name, files=files))
        return cases

    def build_records(
        self,
        *,
        include_excluded: bool = False,
        validate_images: bool = True,
    ) -> list[StandardizedRecord]:
        records: list[StandardizedRecord] = []
        for case in self.discover_cases():
            records.extend(self._build_case_records(case, validate_images=validate_images))
        if include_excluded:
            return records
        return [record for record in records if not record.exclude_reason]

    def write_manifest(
        self,
        output_path: str | Path,
        *,
        include_excluded: bool = False,
        validate_images: bool = True,
    ) -> list[StandardizedRecord]:
        records = self.build_records(
            include_excluded=include_excluded,
            validate_images=validate_images,
        )
        write_standardized_manifest(output_path, records)
        return records

    def _build_case_records(
        self,
        case: UtswGliomaCase,
        *,
        validate_images: bool,
    ) -> list[StandardizedRecord]:
        records = [self._build_classification_record(case, validate_images=validate_images)]
        segmentation_record = self._build_segmentation_record(case, validate_images=validate_images)
        if segmentation_record is not None:
            records.append(segmentation_record)
        return records

    def _build_classification_record(
        self,
        case: UtswGliomaCase,
        *,
        validate_images: bool,
    ) -> StandardizedRecord:
        t1_path = case.get_t1_path()
        exclude_reason = ""
        if t1_path is None:
            exclude_reason = "missing_t1_image"
        elif validate_images:
            exclude_reason = _image_validation_exclude_reason(t1_path)

        return StandardizedRecord(
            dataset_key=UTSW_GLIOMA_DATASET_KEY,
            subject_id=case.subject_key,
            visit_id="baseline",
            global_case_id=case.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UTSW_GLIOMA_PREPROC_PROFILE,
            source_study="UTSW-Glioma",
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _build_segmentation_record(
        self,
        case: UtswGliomaCase,
        *,
        validate_images: bool,
    ) -> StandardizedRecord | None:
        anchor_path = case.get_anchor_path()
        mask_key = case.get_mask_key()
        mask_path = case.get_mask_path()
        if anchor_path is None and mask_path is None:
            return None

        exclude_reason = ""
        if anchor_path is None:
            exclude_reason = "missing_anchor_image"
        elif mask_path is None:
            exclude_reason = "missing_segmentation_mask"
        elif validate_images:
            exclude_reason = _image_validation_exclude_reason(mask_path)
            if exclude_reason == "empty_brain_after_load":
                exclude_reason = "empty_segmentation_mask"

        mask_tier = ""
        if mask_key in {"tumorseg_manual_correction", "rtumorseg_manual_correction"}:
            mask_tier = "curated"
        elif mask_key == "tumorseg_fets":
            mask_tier = "derived"

        t1_path = case.get_t1_path()
        return StandardizedRecord(
            dataset_key=UTSW_GLIOMA_DATASET_KEY,
            subject_id=case.subject_key,
            visit_id="baseline",
            global_case_id=f"{case.global_case_id}__seg",
            image_path="" if anchor_path is None else str(anchor_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UTSW_GLIOMA_SEG_PREPROC_PROFILE,
            source_study="UTSW-Glioma",
            exclude_reason=exclude_reason,
            task_type="segmentation",
            mask_path="" if mask_path is None else str(mask_path),
            mask_tier=mask_tier,
        )
