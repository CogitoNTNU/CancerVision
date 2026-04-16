"""Adapter for Yale Brain Mets Longitudinal NIfTI package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np

from ..constants import (
    YALE_BRAIN_METS_LONGITUDINAL_DATASET_KEY,
    YALE_BRAIN_METS_LONGITUDINAL_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

NIFTI_SUFFIXES = (".nii", ".nii.gz")
YALE_MODALITY_PATTERN = re.compile(
    r"^(?P<subject>YG_[A-Z0-9]+)_(?P<visit>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<modality>[^.]+)$",
    re.IGNORECASE,
)


def _strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def _normalize_modality(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "pre": "t1",
        "post": "t1ce",
        "flair": "flair",
        "t2": "t2",
    }
    return aliases.get(normalized, normalized)


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
class ParsedYaleSeries:
    subject_id: str
    visit_id: str
    acquisition_time: str
    modality: str


@dataclass(slots=True)
class YaleVisitGroup:
    subject_id: str
    visit_id: str
    acquisition_time: str
    files: dict[str, Path]

    @property
    def subject_key(self) -> str:
        return f"{YALE_BRAIN_METS_LONGITUDINAL_DATASET_KEY}:{self.subject_id}"

    @property
    def global_case_id(self) -> str:
        return (
            f"{YALE_BRAIN_METS_LONGITUDINAL_DATASET_KEY}"
            f"__{self.subject_id}__{self.visit_id}"
        )

    def get_t1_path(self) -> Path | None:
        return self.files.get("t1")


def parse_yale_series_path(path: str | Path) -> ParsedYaleSeries | None:
    """Parse Yale filename into subject, visit, acquisition time, modality."""

    source = Path(path)
    stem = _strip_nii_suffix(source.name)
    match = YALE_MODALITY_PATTERN.match(stem)
    if match is None:
        return None
    return ParsedYaleSeries(
        subject_id=match.group("subject"),
        visit_id=match.group("visit"),
        acquisition_time=match.group("time"),
        modality=_normalize_modality(match.group("modality")),
    )


class YaleBrainMetsLongitudinalAdapter:
    """Crawl Yale NIfTI package and emit classification rows."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover_visit_groups(self) -> list[YaleVisitGroup]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Yale Brain Mets root not found: {self.root}")

        grouped: dict[tuple[str, str], YaleVisitGroup] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                continue
            parsed = parse_yale_series_path(path)
            if parsed is None:
                continue
            key = (parsed.subject_id, parsed.visit_id)
            group = grouped.get(key)
            if group is None:
                group = YaleVisitGroup(
                    subject_id=parsed.subject_id,
                    visit_id=parsed.visit_id,
                    acquisition_time=parsed.acquisition_time,
                    files={},
                )
                grouped[key] = group
            group.files.setdefault(parsed.modality, path)

        return [grouped[key] for key in sorted(grouped)]

    def build_records(
        self,
        *,
        include_excluded: bool = False,
        validate_images: bool = True,
    ) -> list[StandardizedRecord]:
        records = [
            self._build_classification_record(group, validate_images=validate_images)
            for group in self.discover_visit_groups()
        ]
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

    def _build_classification_record(
        self,
        group: YaleVisitGroup,
        *,
        validate_images: bool,
    ) -> StandardizedRecord:
        t1_path = group.get_t1_path()
        exclude_reason = ""
        if t1_path is None:
            exclude_reason = "missing_t1_image"
        elif validate_images:
            exclude_reason = _image_validation_exclude_reason(t1_path)

        return StandardizedRecord(
            dataset_key=YALE_BRAIN_METS_LONGITUDINAL_DATASET_KEY,
            subject_id=group.subject_key,
            visit_id=group.visit_id,
            global_case_id=group.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="brain_metastasis",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=YALE_BRAIN_METS_LONGITUDINAL_PREPROC_PROFILE,
            source_study="Yale-Brain-Mets-Longitudinal",
            exclude_reason=exclude_reason,
            task_type="classification",
        )
