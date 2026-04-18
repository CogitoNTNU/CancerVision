"""Adapters for BraTS 2020/2023/2024 NIfTI packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np

from src.datasets.brats_paths import (
    BraTSLayout,
    detect_brats_layout,
    find_nifti,
    resolve_brats_data_dir,
)

from ..io import write_standardized_manifest
from ..models import StandardizedRecord


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
class BraTSDatasetSpec:
    dataset_key: str
    source_study: str
    classification_preproc_profile: str
    segmentation_preproc_profile: str


@dataclass(slots=True)
class BraTSCase:
    dataset_key: str
    patient_id: str
    layout: BraTSLayout
    case_dir: Path

    @property
    def subject_key(self) -> str:
        return f"{self.dataset_key}:{self.patient_id}"

    @property
    def visit_id(self) -> str:
        return "baseline"

    @property
    def global_case_id(self) -> str:
        return f"{self.dataset_key}__{self.patient_id}__{self.visit_id}"

    def _resolve_modality_path(self, modality_suffix: str) -> Path | None:
        try:
            return Path(
                find_nifti(
                    self.case_dir,
                    f"{self.patient_id}{self.layout.separator}{modality_suffix}",
                )
            )
        except FileNotFoundError:
            return None

    def get_t1_path(self) -> Path | None:
        suffix = "t1" if self.layout.name == "brats2020" else "t1n"
        return self._resolve_modality_path(suffix)

    def get_anchor_path(self) -> Path | None:
        suffix = "t1ce" if self.layout.name == "brats2020" else "t1c"
        return self._resolve_modality_path(suffix)

    def get_mask_path(self) -> Path | None:
        return self._resolve_modality_path(self.layout.label_suffix)


class BraTSAdapter:
    """Crawl BraTS patient folders and emit standardized rows."""

    def __init__(self, root: str | Path, *, spec: BraTSDatasetSpec) -> None:
        self.root = Path(root)
        self.spec = spec

    def discover_cases(self) -> list[BraTSCase]:
        resolved_root = resolve_brats_data_dir(self.root)
        cases: list[BraTSCase] = []
        for case_dir in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
            layout = detect_brats_layout(case_dir.name)
            if layout is None:
                continue
            cases.append(
                BraTSCase(
                    dataset_key=self.spec.dataset_key,
                    patient_id=case_dir.name,
                    layout=layout,
                    case_dir=case_dir,
                )
            )
        if not cases:
            raise FileNotFoundError(
                f"No BraTS patient folders found in {resolved_root}"
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
        case: BraTSCase,
        *,
        validate_images: bool,
    ) -> list[StandardizedRecord]:
        records = [
            self._build_classification_record(case, validate_images=validate_images)
        ]
        segmentation_record = self._build_segmentation_record(
            case,
            validate_images=validate_images,
        )
        if segmentation_record is not None:
            records.append(segmentation_record)
        return records

    def _build_classification_record(
        self,
        case: BraTSCase,
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
            dataset_key=self.spec.dataset_key,
            subject_id=case.subject_key,
            visit_id=case.visit_id,
            global_case_id=case.global_case_id,
            image_path="" if t1_path is None else str(t1_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=self.spec.classification_preproc_profile,
            source_study=self.spec.source_study,
            source_split="",
            exclude_reason=exclude_reason,
            task_type="classification",
        )

    def _build_segmentation_record(
        self,
        case: BraTSCase,
        *,
        validate_images: bool,
    ) -> StandardizedRecord:
        anchor_path = case.get_anchor_path()
        mask_path = case.get_mask_path()
        exclude_reason = ""
        if anchor_path is None:
            exclude_reason = "missing_anchor_image"
        elif mask_path is None:
            exclude_reason = "missing_segmentation_mask"
        elif validate_images:
            exclude_reason = _image_validation_exclude_reason(mask_path)
            if exclude_reason == "empty_brain_after_load":
                exclude_reason = "empty_segmentation_mask"

        t1_path = case.get_t1_path()
        return StandardizedRecord(
            dataset_key=self.spec.dataset_key,
            subject_id=case.subject_key,
            visit_id=case.visit_id,
            global_case_id=f"{case.global_case_id}__seg",
            image_path="" if anchor_path is None else str(anchor_path),
            t1_path="" if t1_path is None else str(t1_path),
            diagnosis_original="glioma",
            binary_status="unhealthy",
            label="unhealthy",
            label_family="tumor",
            preproc_profile=self.spec.segmentation_preproc_profile,
            source_study=self.spec.source_study,
            source_split="",
            exclude_reason=exclude_reason,
            task_type="segmentation",
            mask_path="" if mask_path is None else str(mask_path),
            mask_tier="curated" if mask_path is not None else "",
        )
