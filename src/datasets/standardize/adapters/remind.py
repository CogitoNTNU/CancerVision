"""Adapter for ReMIND DICOM images plus NRRD masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pydicom

from ..constants import (
    REMIND_DATASET_KEY,
    REMIND_PREPROC_PROFILE,
    REMIND_SEG_PREPROC_PROFILE,
)
from ..io import write_standardized_manifest
from ..models import StandardizedRecord

REAL_T1_PATTERNS = ("t1_precontrast", "t1_mp2rage")


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _normalize_series_description(value: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", _normalize_text(value).lower())
    return normalized.strip("_")


def _sanitize_suffix(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "unknown"


def is_remind_real_t1_description(series_description: str | None) -> bool:
    normalized = _normalize_series_description(series_description)
    return any(pattern in normalized for pattern in REAL_T1_PATTERNS)


@dataclass(frozen=True, slots=True)
class RemindMaskReference:
    subject_id: str
    phase: str
    label_name: str
    series_description: str
    path: Path

    @property
    def visit_id(self) -> str:
        return self.phase


@dataclass(frozen=True, slots=True)
class RemindSeries:
    subject_id: str
    study_uid: str
    series_uid: str
    series_description: str
    modality: str
    path: Path

    @property
    def normalized_description(self) -> str:
        return _normalize_series_description(self.series_description)


def parse_remind_mask_filename(path: str | Path) -> RemindMaskReference | None:
    """Parse `ReMIND-001-preop-SEG-tumor-MR-3D_AX_T1_postcontrast.nrrd`."""

    source = Path(path)
    stem = source.name
    if stem.endswith(".nrrd"):
        stem = stem[:-5]
    match = re.match(
        r"^(?P<subject>ReMIND-\d+)-(?P<phase>[^-]+)-SEG-(?P<label>.+)-MR-(?P<series>.+)$",
        stem,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return RemindMaskReference(
        subject_id=_normalize_text(match.group("subject")),
        phase=_normalize_text(match.group("phase")).lower(),
        label_name=_normalize_text(match.group("label")).lower(),
        series_description=_normalize_text(match.group("series")),
        path=source,
    )


class RemindAdapter:
    """Crawl ReMIND DICOM series and pair preop tumor masks."""

    def __init__(self, image_root: str | Path, mask_root: str | Path) -> None:
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)

    def discover_series(self) -> dict[str, list[RemindSeries]]:
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"ReMIND image root not found: {self.image_root}")

        subjects: dict[str, list[RemindSeries]] = {}
        for subject_dir in sorted(path for path in self.image_root.iterdir() if path.is_dir()):
            subject_id = subject_dir.name
            subject_series: list[RemindSeries] = []
            for study_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
                for series_dir in sorted(path for path in study_dir.iterdir() if path.is_dir()):
                    first_file = next(
                        (path for path in sorted(series_dir.iterdir()) if path.is_file() and path.suffix.lower() == ".dcm"),
                        None,
                    )
                    if first_file is None:
                        continue
                    dataset = pydicom.dcmread(str(first_file), stop_before_pixels=True)
                    modality = _normalize_text(getattr(dataset, "Modality", ""))
                    if modality != "MR":
                        continue
                    series_description = _normalize_text(
                        getattr(dataset, "SeriesDescription", None)
                    )
                    if not series_description:
                        continue
                    subject_series.append(
                        RemindSeries(
                            subject_id=subject_id,
                            study_uid=study_dir.name,
                            series_uid=series_dir.name,
                            series_description=series_description,
                            modality=modality,
                            path=series_dir,
                        )
                    )
            subjects[subject_id] = subject_series
        return subjects

    def discover_masks(self) -> dict[str, list[RemindMaskReference]]:
        if not self.mask_root.is_dir():
            raise FileNotFoundError(f"ReMIND mask root not found: {self.mask_root}")

        subjects: dict[str, list[RemindMaskReference]] = {}
        for subject_dir in sorted(path for path in self.mask_root.iterdir() if path.is_dir()):
            subject_masks: list[RemindMaskReference] = []
            for mask_path in sorted(subject_dir.glob("*.nrrd")):
                parsed = parse_remind_mask_filename(mask_path)
                if parsed is None:
                    continue
                subject_masks.append(parsed)
            subjects[subject_dir.name] = subject_masks
        return subjects

    def build_records(self, *, include_excluded: bool = False) -> list[StandardizedRecord]:
        series_by_subject = self.discover_series()
        masks_by_subject = self.discover_masks()
        subject_ids = sorted(set(series_by_subject) | set(masks_by_subject))

        records: list[StandardizedRecord] = []
        for subject_id in subject_ids:
            subject_series = series_by_subject.get(subject_id, [])
            subject_masks = masks_by_subject.get(subject_id, [])
            records.extend(self._build_subject_records(subject_id, subject_series, subject_masks))
        if include_excluded:
            return records
        return [record for record in records if not record.exclude_reason]

    def write_manifest(
        self,
        output_path: str | Path,
        *,
        include_excluded: bool = False,
    ) -> list[StandardizedRecord]:
        records = self.build_records(include_excluded=include_excluded)
        write_standardized_manifest(output_path, records)
        return records

    def _build_subject_records(
        self,
        subject_id: str,
        subject_series: list[RemindSeries],
        subject_masks: list[RemindMaskReference],
    ) -> list[StandardizedRecord]:
        records: list[StandardizedRecord] = []
        classification_series = next(
            (
                series
                for series in sorted(subject_series, key=lambda item: item.series_description)
                if is_remind_real_t1_description(series.series_description)
            ),
            None,
        )

        records.append(
            StandardizedRecord(
                dataset_key=REMIND_DATASET_KEY,
                subject_id=f"{REMIND_DATASET_KEY}:{subject_id}",
                visit_id="preop",
                global_case_id=f"{REMIND_DATASET_KEY}__{subject_id}__preop",
                image_path="" if classification_series is None else str(classification_series.path),
                t1_path="" if classification_series is None else str(classification_series.path),
                diagnosis_original="glioma",
                binary_status="unhealthy",
                label="unhealthy",
                label_family="tumor",
                preproc_profile=REMIND_PREPROC_PROFILE,
                source_study="ReMIND",
                exclude_reason="" if classification_series is not None else "missing_t1_image",
                task_type="classification",
            )
        )

        series_lookup: dict[str, list[RemindSeries]] = {}
        for series in subject_series:
            series_lookup.setdefault(series.normalized_description, []).append(series)

        for mask in subject_masks:
            if mask.phase != "preop" or mask.label_name != "tumor":
                continue
            matched_series = series_lookup.get(
                _normalize_series_description(mask.series_description),
                [],
            )
            if len(matched_series) == 1:
                matched_path = str(matched_series[0].path)
                exclude_reason = ""
            elif len(matched_series) == 0:
                matched_path = ""
                exclude_reason = "missing_series_for_mask"
            else:
                matched_path = ""
                exclude_reason = "ambiguous_mask_series_match"

            records.append(
                StandardizedRecord(
                    dataset_key=REMIND_DATASET_KEY,
                    subject_id=f"{REMIND_DATASET_KEY}:{subject_id}",
                    visit_id=mask.visit_id,
                    global_case_id=(
                        f"{REMIND_DATASET_KEY}__{subject_id}__{mask.visit_id}__seg__"
                        f"{_sanitize_suffix(mask.series_description)}"
                    ),
                    image_path=matched_path,
                    t1_path="" if classification_series is None else str(classification_series.path),
                    diagnosis_original="glioma",
                    binary_status="unhealthy",
                    label="unhealthy",
                    label_family="tumor",
                    preproc_profile=REMIND_SEG_PREPROC_PROFILE,
                    source_study="ReMIND",
                    exclude_reason=exclude_reason,
                    task_type="segmentation",
                    mask_path=str(mask.path),
                    mask_tier="curated",
                )
            )
        return records

