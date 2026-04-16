"""Adapter for `radiata-ai/brain-structure` metadata-driven ingestion."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from ..constants import (
    BRAIN_STRUCTURE_DATASET_KEY,
    BRAIN_STRUCTURE_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

REQUIRED_METADATA_COLUMNS = {
    "t1_local_path",
    "split",
    "study",
    "participant_id",
    "session_id",
    "age",
    "sex",
    "clinical_diagnosis",
    "scanner_manufacturer",
    "scanner_model",
    "field_strength",
    "image_quality_rating",
    "total_intracranial_volume",
    "radiata_id",
}

IGNORED_IMAGE_SOURCES = {
    "data.zip",
    "sample_images",
    "README.md",
    "brain-structure.py",
}

HEALTHY_DIAGNOSES = {
    "cognitively_normal",
    "cn",
    "normal_control",
    "healthy_control",
    "control",
    "healthy",
}

ALZHEIMER_DIAGNOSES = {
    "ad",
    "alzheimers_disease",
    "alzheimer_disease",
    "probable_alzheimers_disease",
    "probable_alzheimer_disease",
    "mci_due_to_alzheimers_disease",
    "mci_due_to_alzheimer_disease",
}


@dataclass(frozen=True, slots=True)
class DiagnosisMapping:
    """Mapped diagnosis values used for downstream task manifests."""

    label: str
    label_family: str
    binary_status: str


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _normalize_diagnosis(value: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", _normalize_text(value).lower())
    return normalized.strip("_")


def map_brain_structure_diagnosis(raw_value: str | None) -> DiagnosisMapping | None:
    """Map source diagnosis into project labels."""

    normalized = _normalize_diagnosis(raw_value)
    if not normalized:
        return None

    if normalized in HEALTHY_DIAGNOSES:
        return DiagnosisMapping(
            label="healthy",
            label_family="healthy_control",
            binary_status="healthy",
        )

    if normalized in ALZHEIMER_DIAGNOSES or "alzheimer" in normalized:
        return DiagnosisMapping(
            label="unhealthy",
            label_family="neurodegenerative",
            binary_status="unhealthy",
        )

    tokens = {token for token in normalized.split("_") if token}
    ad_tokens = {"ad", "alzheimers", "alzheimer"}
    if tokens & ad_tokens:
        return DiagnosisMapping(
            label="unhealthy",
            label_family="neurodegenerative",
            binary_status="unhealthy",
        )

    return None


def _metadata_relative_path(path_text: str) -> Path:
    sanitized = _normalize_text(path_text).replace("\\", "/").lstrip("/")
    return Path(sanitized)


def _has_nonzero_brain(image_path: Path) -> bool:
    data = np.asanyarray(nib.load(str(image_path)).dataobj)
    return bool(np.any(data != 0))


class BrainStructureAdapter:
    """Read `brain-structure` rows from metadata.csv and emit manifest rows."""

    def __init__(self, root: str | Path, metadata_filename: str = "metadata.csv") -> None:
        self.root = Path(root)
        self.metadata_path = self.root / metadata_filename

    def load_metadata_rows(self) -> list[dict[str, str]]:
        """Load metadata rows after validating required schema."""

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_path}")

        with self.metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_METADATA_COLUMNS - fieldnames)
            if missing:
                raise ValueError(
                    "brain-structure metadata missing required columns: "
                    + ", ".join(missing)
                )

            return [
                {key: _normalize_text(value) for key, value in row.items()}
                for row in reader
            ]

    def build_records(
        self,
        *,
        include_excluded: bool = False,
        validate_images: bool = True,
    ) -> list[StandardizedRecord]:
        """Build standardized manifest rows from metadata."""

        records = [
            self._build_record(row, validate_images=validate_images)
            for row in self.load_metadata_rows()
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
        """Build and write standardized manifest CSV."""

        records = self.build_records(
            include_excluded=include_excluded,
            validate_images=validate_images,
        )
        write_standardized_manifest(output_path, records)
        return records

    def _build_record(
        self,
        row: dict[str, str],
        *,
        validate_images: bool,
    ) -> StandardizedRecord:
        relative_path = _metadata_relative_path(row["t1_local_path"])
        image_path = self.root / relative_path
        diagnosis_original = row["clinical_diagnosis"]
        mapped_diagnosis = map_brain_structure_diagnosis(diagnosis_original)
        exclude_reason = self._exclude_reason_for_row(
            image_path=image_path,
            diagnosis=mapped_diagnosis,
            validate_images=validate_images,
        )

        return StandardizedRecord(
            dataset_key=BRAIN_STRUCTURE_DATASET_KEY,
            subject_id=(
                f"{BRAIN_STRUCTURE_DATASET_KEY}:{row['study']}:{row['participant_id']}"
            ),
            visit_id=row["session_id"],
            global_case_id=(
                f"{BRAIN_STRUCTURE_DATASET_KEY}__{row['study']}__"
                f"{row['participant_id']}__{row['session_id']}"
            ),
            image_path=str(image_path),
            t1_path=str(image_path),
            diagnosis_original=diagnosis_original,
            binary_status="" if mapped_diagnosis is None else mapped_diagnosis.binary_status,
            label="" if mapped_diagnosis is None else mapped_diagnosis.label,
            label_family="" if mapped_diagnosis is None else mapped_diagnosis.label_family,
            preproc_profile=BRAIN_STRUCTURE_PREPROC_PROFILE,
            source_study=row["study"],
            source_split=row["split"],
            radiata_id=row["radiata_id"],
            image_quality_rating=row["image_quality_rating"],
            total_intracranial_volume=row["total_intracranial_volume"],
            age=row["age"],
            sex=row["sex"],
            scanner_manufacturer=row["scanner_manufacturer"],
            scanner_model=row["scanner_model"],
            field_strength=row["field_strength"],
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _exclude_reason_for_row(
        self,
        *,
        image_path: Path,
        diagnosis: DiagnosisMapping | None,
        validate_images: bool,
    ) -> str:
        if any(part in IGNORED_IMAGE_SOURCES for part in image_path.parts):
            return "missing_image_file"

        if not image_path.is_file():
            if (self.root / "data.zip").is_file():
                return "zip_only_not_extracted"
            return "missing_image_file"

        if diagnosis is None:
            return "unmapped_brain_structure_label"

        if validate_images and not _has_nonzero_brain(image_path):
            return "empty_brain_after_load"

        return ""
