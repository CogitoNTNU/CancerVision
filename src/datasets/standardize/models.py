"""Shared manifest models for dataset standardization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True, slots=True)
class StandardizedRecord:
    """Normalized manifest row shared across datasets."""

    dataset_key: str
    subject_id: str
    visit_id: str
    global_case_id: str
    image_path: str
    t1_path: str
    diagnosis_original: str
    binary_status: str
    label: str
    label_family: str
    preproc_profile: str
    source_study: str = ""
    source_split: str = ""
    radiata_id: str = ""
    image_quality_rating: str = ""
    total_intracranial_volume: str = ""
    age: str = ""
    sex: str = ""
    scanner_manufacturer: str = ""
    scanner_model: str = ""
    field_strength: str = ""
    exclude_reason: str = ""
    task_type: str = "classification"
    mask_path: str = ""
    brain_mask_path: str = ""
    normalization_mask_method: str = ""
    mask_tier: str = ""

    def to_row(self) -> dict[str, str]:
        """Render record as CSV-safe string mapping."""

        return {
            key: "" if value is None else str(value)
            for key, value in asdict(self).items()
        }


def standardized_manifest_fieldnames() -> list[str]:
    """Return stable field order for standardized manifests."""

    return [field.name for field in fields(StandardizedRecord)]
