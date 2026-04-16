"""Adapter for UCSD-PTGBM case folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np

from ..constants import (
    UCSD_PTGBM_DATASET_KEY,
    UCSD_PTGBM_PREPROC_PROFILE,
    UCSD_PTGBM_SEG_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

NIFTI_SUFFIXES = (".nii", ".nii.gz")
STRUCTURAL_ANCHOR_PRIORITY = ("t1ce", "t1", "t2", "flair")
MASK_PRIORITY = (
    "brats_tumor_seg",
    "total_cellular_tumor_seg",
    "enhancing_cellular_tumor_seg",
    "non_enhancing_cellular_tumor_seg",
)
MODALITY_ALIASES = {
    "t1pre": "t1",
    "t1post": "t1ce",
    "t2": "t2",
    "flair": "flair",
    "brats_tumor_seg": "brats_tumor_seg",
    "total_cellular_tumor_seg": "total_cellular_tumor_seg",
    "enhancing_cellular_tumor_seg": "enhancing_cellular_tumor_seg",
    "non_enhancing_cellular_tumor_seg": "non_enhancing_cellular_tumor_seg",
}


def _strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _normalize_modality(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return MODALITY_ALIASES.get(normalized, normalized)


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


@dataclass(frozen=True, slots=True)
class ParsedUcsdCase:
    subject_id: str
    visit_id: str
    modality: str


@dataclass(slots=True)
class UcsdPtgbmCase:
    subject_id: str
    visit_id: str
    files: dict[str, Path]

    @property
    def subject_key(self) -> str:
        return f"{UCSD_PTGBM_DATASET_KEY}:{self.subject_id}"

    @property
    def global_case_id(self) -> str:
        return f"{UCSD_PTGBM_DATASET_KEY}__{self.subject_id}__{self.visit_id}"

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


def parse_ucsd_case_dir_name(name: str) -> tuple[str, str] | None:
    """Parse `UCSD-PTGBM-0002_01` into subject and visit."""

    match = re.match(r"^(?P<subject>UCSD-PTGBM-\d+)_(?P<visit>[^_]+)$", name)
    if match is None:
        return None
    return _normalize_text(match.group("subject")), _normalize_text(match.group("visit"))


def parse_ucsd_series_name(filename: str, *, case_prefix: str) -> str | None:
    """Parse case-local filename into modality key."""

    stem = _strip_nii_suffix(filename)
    if not stem.startswith(case_prefix):
        return None
    suffix = stem[len(case_prefix) :].lstrip("_")
    if not suffix:
        return None
    return _normalize_modality(suffix)


class UcsdPtgbmAdapter:
    """Crawl UCSD-PTGBM case folders and emit standardized rows."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover_cases(self) -> list[UcsdPtgbmCase]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"UCSD-PTGBM root not found: {self.root}")

        cases: list[UcsdPtgbmCase] = []
        for case_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            parsed = parse_ucsd_case_dir_name(case_dir.name)
            if parsed is None:
                continue
            subject_id, visit_id = parsed
            files: dict[str, Path] = {}
            for path in sorted(case_dir.iterdir()):
                if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                    continue
                modality = parse_ucsd_series_name(path.name, case_prefix=case_dir.name)
                if modality is None:
                    continue
                files.setdefault(modality, path)
            if files:
                cases.append(
                    UcsdPtgbmCase(
                        subject_id=subject_id,
                        visit_id=visit_id,
                        files=files,
                    )
                )
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
        case: UcsdPtgbmCase,
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
        case: UcsdPtgbmCase,
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
            dataset_key=UCSD_PTGBM_DATASET_KEY,
            subject_id=case.subject_key,
            visit_id=case.visit_id,
            global_case_id=case.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioblastoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UCSD_PTGBM_PREPROC_PROFILE,
            source_study="UCSD-PTGBM",
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _build_segmentation_record(
        self,
        case: UcsdPtgbmCase,
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

        mask_tier = "curated" if mask_key == "brats_tumor_seg" else "derived"

        t1_path = case.get_t1_path()
        return StandardizedRecord(
            dataset_key=UCSD_PTGBM_DATASET_KEY,
            subject_id=case.subject_key,
            visit_id=case.visit_id,
            global_case_id=f"{case.global_case_id}__seg",
            image_path="" if anchor_path is None else str(anchor_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioblastoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UCSD_PTGBM_SEG_PREPROC_PROFILE,
            source_study="UCSD-PTGBM",
            exclude_reason=exclude_reason,
            task_type="segmentation",
            mask_path="" if mask_path is None else str(mask_path),
            mask_tier=mask_tier,
        )
