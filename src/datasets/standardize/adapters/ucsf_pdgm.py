"""Adapter for `UCSF-PDGM-v5` folder crawls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from ..constants import (
    UCSF_PDGM_DATASET_KEY,
    UCSF_PDGM_PREPROC_PROFILE,
    UCSF_PDGM_SEG_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

NIFTI_SUFFIXES = (".nii", ".nii.gz")
CASE_DIR_SUFFIX = "_nifti"

STRUCTURAL_ANCHOR_PRIORITY = ("t1", "t1ce", "t2", "flair")
MASK_TOKENS = {
    "tumor_segmentation": "tumor_segmentation",
}
MODALITY_ALIASES = {
    "t1": "t1",
    "t1_bias": "t1_bias",
    "t1c": "t1ce",
    "t1c_bias": "t1ce_bias",
    "t2": "t2",
    "t2_bias": "t2_bias",
    "flair": "flair",
    "flair_bias": "flair_bias",
}


def _strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def _has_nonzero_volume(image_path: Path) -> bool:
    data = np.asanyarray(nib.load(str(image_path)).dataobj)
    return bool(np.any(data != 0))


@dataclass(slots=True)
class UcsfPdgmCase:
    subject_id: str
    visit_id: str
    files: dict[str, Path]

    @property
    def subject_key(self) -> str:
        return f"{UCSF_PDGM_DATASET_KEY}:{self.subject_id}"

    @property
    def global_case_id(self) -> str:
        return f"{UCSF_PDGM_DATASET_KEY}__{self.subject_id}__{self.visit_id}"

    def get_t1_path(self) -> Path | None:
        return self.files.get("t1")

    def get_anchor_path(self) -> Path | None:
        for modality in STRUCTURAL_ANCHOR_PRIORITY:
            path = self.files.get(modality)
            if path is not None:
                return path
        return None

    def get_mask_path(self) -> Path | None:
        return self.files.get("tumor_segmentation")


def parse_ucsf_case_dir_name(name: str) -> tuple[str, str] | None:
    """Parse `UCSF-PDGM-0431_FU001d_nifti` into subject and visit."""

    if not name.endswith(CASE_DIR_SUFFIX):
        return None
    stem = name[: -len(CASE_DIR_SUFFIX)]
    if stem.endswith("_"):
        stem = stem[:-1]
    match = re.match(r"^(?P<subject>UCSF-PDGM-\d+)(?:_(?P<visit>.+))?$", stem)
    if match is None:
        return None
    subject_id = match.group("subject")
    visit_id = match.group("visit") or "baseline"
    return subject_id, visit_id


def parse_ucsf_series_name(filename: str, *, case_prefix: str) -> str | None:
    """Parse series file modality from case-relative filename."""

    stem = _strip_nii_suffix(filename)
    if not stem.startswith(case_prefix):
        return None
    suffix = stem[len(case_prefix) :].lstrip("_")
    if not suffix:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", suffix.lower()).strip("_")
    if normalized in MASK_TOKENS:
        return MASK_TOKENS[normalized]
    return MODALITY_ALIASES.get(normalized)


class UcsfPdgmAdapter:
    """Crawl NIfTI case folders and emit standardized UCSF-PDGM rows."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover_cases(self) -> list[UcsfPdgmCase]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"UCSF-PDGM root not found: {self.root}")

        cases: list[UcsfPdgmCase] = []
        for case_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            parsed = parse_ucsf_case_dir_name(case_dir.name)
            if parsed is None:
                continue
            subject_id, visit_id = parsed
            case_prefix = case_dir.name[: -len(CASE_DIR_SUFFIX)].rstrip("_")
            files: dict[str, Path] = {}
            for path in sorted(case_dir.iterdir()):
                if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                    continue
                modality = parse_ucsf_series_name(path.name, case_prefix=case_prefix)
                if modality is None:
                    continue
                files.setdefault(modality, path)
            cases.append(UcsfPdgmCase(subject_id=subject_id, visit_id=visit_id, files=files))
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
        case: UcsfPdgmCase,
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
        case: UcsfPdgmCase,
        *,
        validate_images: bool,
    ) -> StandardizedRecord:
        t1_path = case.get_t1_path()
        exclude_reason = ""
        if t1_path is None:
            exclude_reason = "missing_t1_image"
        elif validate_images and not _has_nonzero_volume(t1_path):
            exclude_reason = "empty_brain_after_load"

        return StandardizedRecord(
            dataset_key=UCSF_PDGM_DATASET_KEY,
            subject_id=case.subject_key,
            visit_id=case.visit_id,
            global_case_id=case.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UCSF_PDGM_PREPROC_PROFILE,
            source_study="UCSF-PDGM",
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _build_segmentation_record(
        self,
        case: UcsfPdgmCase,
        *,
        validate_images: bool,
    ) -> StandardizedRecord | None:
        anchor_path = case.get_anchor_path()
        mask_path = case.get_mask_path()
        if anchor_path is None and mask_path is None:
            return None

        exclude_reason = ""
        if anchor_path is None:
            exclude_reason = "missing_anchor_image"
        elif mask_path is None:
            exclude_reason = "missing_segmentation_mask"
        elif validate_images and not _has_nonzero_volume(mask_path):
            exclude_reason = "empty_segmentation_mask"

        t1_path = case.get_t1_path()
        return StandardizedRecord(
            dataset_key=UCSF_PDGM_DATASET_KEY,
            subject_id=case.subject_key,
            visit_id=case.visit_id,
            global_case_id=f"{case.global_case_id}__seg",
            image_path="" if anchor_path is None else str(anchor_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UCSF_PDGM_SEG_PREPROC_PROFILE,
            source_study="UCSF-PDGM",
            exclude_reason=exclude_reason,
            task_type="segmentation",
            mask_path="" if mask_path is None else str(mask_path),
            mask_tier="curated" if mask_path is not None else "",
        )

