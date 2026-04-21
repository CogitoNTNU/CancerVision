"""Adapter for Vestibular-Schwannoma-MC-RC2 NIfTI package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np

from ..constants import (
    VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY,
    VESTIBULAR_SCHWANNOMA_MC_RC2_PREPROC_PROFILE,
    VESTIBULAR_SCHWANNOMA_MC_RC2_SEG_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

NIFTI_SUFFIXES = (".nii", ".nii.gz")
VESTIBULAR_MODALITY_PATTERN = re.compile(
    r"^(?P<subject>VS_MC_RC2_\d+)_(?P<visit>\d{4}-\d{2}-\d{2})_(?P<modality>[^.]+)$",
    re.IGNORECASE,
)
STRUCTURAL_ANCHOR_PRIORITY = ("t1ce", "t1", "t2")


def _strip_nii_suffix(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def _normalize_modality(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "t1": "t1",
        "t1c": "t1ce",
        "t1c_seg": "seg",
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
class ParsedVestibularSeries:
    subject_id: str
    visit_id: str
    modality: str


@dataclass(slots=True)
class VestibularVisitGroup:
    subject_id: str
    visit_id: str
    files: dict[str, Path]

    @property
    def subject_key(self) -> str:
        return f"{VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY}:{self.subject_id}"

    @property
    def global_case_id(self) -> str:
        return (
            f"{VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY}"
            f"__{self.subject_id}__{self.visit_id}"
        )

    def get_t1_path(self) -> Path | None:
        return self.files.get("t1")

    def get_anchor_path(self) -> Path | None:
        for modality in STRUCTURAL_ANCHOR_PRIORITY:
            path = self.files.get(modality)
            if path is not None:
                return path
        return None

    def get_mask_path(self) -> Path | None:
        return self.files.get("seg")


def parse_vestibular_series_path(path: str | Path) -> ParsedVestibularSeries | None:
    """Parse Vestibular Schwannoma filename into subject, visit, modality."""

    source = Path(path)
    stem = _strip_nii_suffix(source.name)
    match = VESTIBULAR_MODALITY_PATTERN.match(stem)
    if match is None:
        return None
    return ParsedVestibularSeries(
        subject_id=match.group("subject"),
        visit_id=match.group("visit"),
        modality=_normalize_modality(match.group("modality")),
    )


class VestibularSchwannomaMcRc2Adapter:
    """Crawl Vestibular Schwannoma NIfTI package and emit standardized rows."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def discover_visit_groups(self) -> list[VestibularVisitGroup]:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Vestibular Schwannoma root not found: {self.root}"
            )

        grouped: dict[tuple[str, str], VestibularVisitGroup] = {}
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or not path.name.endswith(NIFTI_SUFFIXES):
                continue
            parsed = parse_vestibular_series_path(path)
            if parsed is None:
                continue
            key = (parsed.subject_id, parsed.visit_id)
            group = grouped.get(key)
            if group is None:
                group = VestibularVisitGroup(
                    subject_id=parsed.subject_id,
                    visit_id=parsed.visit_id,
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
        records: list[StandardizedRecord] = []
        for group in self.discover_visit_groups():
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
        group: VestibularVisitGroup,
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
        group: VestibularVisitGroup,
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
            dataset_key=VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY,
            subject_id=group.subject_key,
            visit_id=group.visit_id,
            global_case_id=group.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="vestibular_schwannoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=VESTIBULAR_SCHWANNOMA_MC_RC2_PREPROC_PROFILE,
            source_study="Vestibular-Schwannoma-MC-RC2",
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _build_segmentation_record(
        self,
        group: VestibularVisitGroup,
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
        elif validate_images:
            exclude_reason = _image_validation_exclude_reason(mask_path)
            if exclude_reason == "empty_brain_after_load":
                exclude_reason = "empty_segmentation_mask"

        t1_path = group.get_t1_path()
        return StandardizedRecord(
            dataset_key=VESTIBULAR_SCHWANNOMA_MC_RC2_DATASET_KEY,
            subject_id=group.subject_key,
            visit_id=group.visit_id,
            global_case_id=f"{group.global_case_id}__seg",
            image_path="" if anchor_path is None else str(anchor_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="vestibular_schwannoma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=VESTIBULAR_SCHWANNOMA_MC_RC2_SEG_PREPROC_PROFILE,
            source_study="Vestibular-Schwannoma-MC-RC2",
            exclude_reason=exclude_reason,
            task_type="segmentation",
            mask_path="" if mask_path is None else str(mask_path),
            mask_tier="curated" if mask_path is not None else "",
        )
