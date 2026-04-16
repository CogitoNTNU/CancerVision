"""Adapter for CFB-GBM visit folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np

from ..constants import (
    CFB_GBM_DATASET_KEY,
    CFB_GBM_PREPROC_PROFILE,
    CFB_GBM_SEG_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

NIFTI_SUFFIXES = (".nii", ".nii.gz")
SERIES_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<visit>t\d+)_(?P<modality>[^.]+)\.nii(?:\.gz)?$",
    re.IGNORECASE,
)
STRUCTURAL_ANCHOR_PRIORITY = ("t1ce", "t1", "flair", "t2star", "t2")
REAL_T1_PRIORITY = ("t1eg", "t1tse")
MODALITY_ALIASES = {
    "t1eg": "t1eg",
    "t1tse": "t1tse",
    "t1gd": "t1ce",
    "flair": "flair",
    "t2star": "t2star",
    "t2tse": "t2",
    "gtv": "gtv",
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


@dataclass(frozen=True, slots=True)
class ParsedCfbSeries:
    subject_id: str
    visit_id: str
    modality: str


@dataclass(slots=True)
class CfbGbmVisit:
    subject_id: str
    visit_id: str
    files: dict[str, Path]

    @property
    def subject_key(self) -> str:
        return f"{CFB_GBM_DATASET_KEY}:{self.subject_id}"

    @property
    def global_case_id(self) -> str:
        return f"{CFB_GBM_DATASET_KEY}__{self.subject_id}__{self.visit_id}"

    def get_t1_path(self) -> Path | None:
        for modality in REAL_T1_PRIORITY:
            path = self.files.get(modality)
            if path is not None:
                return path
        return None

    def get_anchor_path(self) -> Path | None:
        for modality in STRUCTURAL_ANCHOR_PRIORITY:
            path = self.files.get(modality)
            if path is not None:
                return path
        return None

    def get_mask_path(self) -> Path | None:
        return self.files.get("gtv")


def parse_cfb_series_path(path: str | Path) -> ParsedCfbSeries | None:
    """Parse CFB-GBM filename into subject, visit, modality."""

    source = Path(path)
    match = SERIES_PATTERN.match(source.name)
    if match is None:
        return None
    raw_modality = match.group("modality").lower()
    modality = MODALITY_ALIASES.get(raw_modality, raw_modality)
    return ParsedCfbSeries(
        subject_id=match.group("subject").zfill(3),
        visit_id=match.group("visit").lower(),
        modality=modality,
    )


class CfbGbmAdapter:
    """Crawl CFB-GBM subject/visit folders and emit standardized rows."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover_visits(self) -> list[CfbGbmVisit]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"CFB-GBM root not found: {self.root}")

        grouped: dict[tuple[str, str], dict[str, Path]] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                continue
            parsed = parse_cfb_series_path(path)
            if parsed is None:
                continue
            key = (parsed.subject_id, parsed.visit_id)
            grouped.setdefault(key, {}).setdefault(parsed.modality, path)

        return [
            CfbGbmVisit(subject_id=subject_id, visit_id=visit_id, files=files)
            for (subject_id, visit_id), files in sorted(grouped.items())
        ]

    def build_records(
        self,
        *,
        include_excluded: bool = False,
        validate_images: bool = True,
    ) -> list[StandardizedRecord]:
        records: list[StandardizedRecord] = []
        for visit in self.discover_visits():
            records.extend(self._build_visit_records(visit, validate_images=validate_images))
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

    def _build_visit_records(
        self,
        visit: CfbGbmVisit,
        *,
        validate_images: bool,
    ) -> list[StandardizedRecord]:
        records = [self._build_classification_record(visit, validate_images=validate_images)]
        segmentation_record = self._build_segmentation_record(
            visit,
            validate_images=validate_images,
        )
        if segmentation_record is not None:
            records.append(segmentation_record)
        return records

    def _build_classification_record(
        self,
        visit: CfbGbmVisit,
        *,
        validate_images: bool,
    ) -> StandardizedRecord:
        t1_path = visit.get_t1_path()
        exclude_reason = ""
        if t1_path is None:
            exclude_reason = "missing_t1_image"
        elif validate_images:
            exclude_reason = _image_validation_exclude_reason(t1_path)

        return StandardizedRecord(
            dataset_key=CFB_GBM_DATASET_KEY,
            subject_id=visit.subject_key,
            visit_id=visit.visit_id,
            global_case_id=visit.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioblastoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=CFB_GBM_PREPROC_PROFILE,
            source_study="CFB-GBM",
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _build_segmentation_record(
        self,
        visit: CfbGbmVisit,
        *,
        validate_images: bool,
    ) -> StandardizedRecord | None:
        anchor_path = visit.get_anchor_path()
        mask_path = visit.get_mask_path()
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

        t1_path = visit.get_t1_path()
        return StandardizedRecord(
            dataset_key=CFB_GBM_DATASET_KEY,
            subject_id=visit.subject_key,
            visit_id=visit.visit_id,
            global_case_id=f"{visit.global_case_id}__seg",
            image_path="" if anchor_path is None else str(anchor_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioblastoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=CFB_GBM_SEG_PREPROC_PROFILE,
            source_study="CFB-GBM",
            exclude_reason=exclude_reason,
            task_type="segmentation",
            mask_path="" if mask_path is None else str(mask_path),
            mask_tier="curated" if mask_path is not None else "",
        )
