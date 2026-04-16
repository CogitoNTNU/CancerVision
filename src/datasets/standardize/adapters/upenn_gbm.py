"""Adapter for TCIA `PKG - UPENN-GBM-NIfTI` folder crawls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from ..constants import (
    UPENN_GBM_DATASET_KEY,
    UPENN_GBM_PREPROC_PROFILE,
    UPENN_GBM_SEG_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

NIFTI_SUFFIXES = (".nii", ".nii.gz")
UPENN_MODALITY_PATTERN = re.compile(
    r"^(?P<subject>.+?)_(?P<visit>[^_]+)_(?P<modality>[^.]+)$",
    re.IGNORECASE,
)

STRUCTURAL_ANCHOR_PRIORITY = ("t1", "t1ce", "t2", "flair")


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _normalize_modality(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "t1": "t1",
        "t1w": "t1",
        "t1_pre": "t1",
        "t1gd": "t1ce",
        "t1_gd": "t1ce",
        "t1ce": "t1ce",
        "t1_ce": "t1ce",
        "t1post": "t1ce",
        "t1_post": "t1ce",
        "t1c": "t1ce",
        "t1_gdce": "t1ce",
        "t2": "t2",
        "flair": "flair",
        "seg": "seg",
        "segmentation": "seg",
        "label": "seg",
        "labels": "seg",
    }
    return aliases.get(normalized, normalized)


def _strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def _has_nonzero_volume(image_path: Path) -> bool:
    data = np.asanyarray(nib.load(str(image_path)).dataobj)
    return bool(np.any(data != 0))


@dataclass(frozen=True, slots=True)
class ParsedSeries:
    subject_id: str
    visit_id: str
    modality: str


@dataclass(slots=True)
class SeriesGroup:
    subject_id: str
    visit_id: str
    files: dict[str, Path]

    @property
    def global_case_id(self) -> str:
        return f"{UPENN_GBM_DATASET_KEY}__{self.subject_id}__{self.visit_id}"

    @property
    def subject_key(self) -> str:
        return f"{UPENN_GBM_DATASET_KEY}:{self.subject_id}"

    def get_anchor_path(self) -> Path | None:
        for modality in STRUCTURAL_ANCHOR_PRIORITY:
            path = self.files.get(modality)
            if path is not None:
                return path
        return None

    def get_t1_path(self) -> Path | None:
        return self.files.get("t1")

    def get_mask_path(self) -> Path | None:
        return self.files.get("seg")


def parse_upenn_series_path(path: str | Path) -> ParsedSeries | None:
    """Parse `SUBJECT_VISIT_MODALITY.nii.gz` into subject/visit/modality."""

    source = Path(path)
    stem = _strip_nii_suffix(source.name)
    match = UPENN_MODALITY_PATTERN.match(stem)
    if match:
        return ParsedSeries(
            subject_id=_normalize_text(match.group("subject")),
            visit_id=_normalize_text(match.group("visit")),
            modality=_normalize_modality(match.group("modality")),
        )

    parent_name = source.parent.name
    parent_match = UPENN_MODALITY_PATTERN.match(parent_name)
    if parent_match:
        return ParsedSeries(
            subject_id=_normalize_text(parent_match.group("subject")),
            visit_id=_normalize_text(parent_match.group("visit")),
            modality=_normalize_modality(stem),
        )
    return None


class UpennGbmAdapter:
    """Crawl NIfTI files and emit standardized UPENN-GBM manifest rows."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover_series_groups(self) -> list[SeriesGroup]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"UPENN-GBM root not found: {self.root}")

        grouped: dict[tuple[str, str], dict[str, Path]] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                continue
            parsed = parse_upenn_series_path(path)
            if parsed is None:
                continue
            key = (parsed.subject_id, parsed.visit_id)
            modalities = grouped.setdefault(key, {})
            modalities.setdefault(parsed.modality, path)

        return [
            SeriesGroup(subject_id=subject_id, visit_id=visit_id, files=files)
            for (subject_id, visit_id), files in sorted(grouped.items())
        ]

    def build_records(
        self,
        *,
        include_excluded: bool = False,
        validate_images: bool = True,
    ) -> list[StandardizedRecord]:
        records: list[StandardizedRecord] = []
        for group in self.discover_series_groups():
            records.extend(
                self._build_group_records(group, validate_images=validate_images)
            )
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

    def _build_group_records(
        self,
        group: SeriesGroup,
        *,
        validate_images: bool,
    ) -> list[StandardizedRecord]:
        records = [
            self._build_classification_record(group, validate_images=validate_images)
        ]
        segmentation_record = self._build_segmentation_record(
            group,
            validate_images=validate_images,
        )
        if segmentation_record is not None:
            records.append(segmentation_record)
        return records

    def _build_classification_record(
        self,
        group: SeriesGroup,
        *,
        validate_images: bool,
    ) -> StandardizedRecord:
        t1_path = group.get_t1_path()
        exclude_reason = ""
        if t1_path is None:
            exclude_reason = "missing_t1_image"
        elif validate_images and not _has_nonzero_volume(t1_path):
            exclude_reason = "empty_brain_after_load"

        return StandardizedRecord(
            dataset_key=UPENN_GBM_DATASET_KEY,
            subject_id=group.subject_key,
            visit_id=group.visit_id,
            global_case_id=group.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioblastoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UPENN_GBM_PREPROC_PROFILE,
            source_study="UPENN-GBM",
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _build_segmentation_record(
        self,
        group: SeriesGroup,
        *,
        validate_images: bool,
    ) -> StandardizedRecord | None:
        anchor_path = group.get_anchor_path()
        mask_path = group.get_mask_path()
        if anchor_path is None and mask_path is None:
            return None

        exclude_reason = ""
        if anchor_path is None:
            exclude_reason = "missing_anchor_image"
        elif mask_path is None:
            exclude_reason = "missing_segmentation_mask"
        elif validate_images and not _has_nonzero_volume(mask_path):
            exclude_reason = "empty_segmentation_mask"

        t1_path = group.get_t1_path()
        return StandardizedRecord(
            dataset_key=UPENN_GBM_DATASET_KEY,
            subject_id=group.subject_key,
            visit_id=group.visit_id,
            global_case_id=f"{group.global_case_id}__seg",
            image_path="" if anchor_path is None else str(anchor_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioblastoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=UPENN_GBM_SEG_PREPROC_PROFILE,
            source_study="UPENN-GBM",
            exclude_reason=exclude_reason,
            task_type="segmentation",
            mask_path="" if mask_path is None else str(mask_path),
            mask_tier="curated" if mask_path is not None else "",
        )

