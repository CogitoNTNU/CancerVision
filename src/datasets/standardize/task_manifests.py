"""Task manifest builders for mixed MRI classification and segmentation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from .constants import BRAIN_STRUCTURE_DATASET_KEY
from .io import write_csv_rows
from .models import standardized_manifest_fieldnames

TASK_MANIFEST_FIELDNAMES = [
    "task_name",
    "task_label",
    "class_name",
    "task_split",
    *standardized_manifest_fieldnames(),
]


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _is_excluded(row: dict[str, object]) -> bool:
    return bool(_text(row.get("exclude_reason")))


def _has_real_t1(row: dict[str, object]) -> bool:
    path = _text(row.get("t1_path"))
    return bool(path) and path.endswith((".nii", ".nii.gz"))


def _is_brain_structure(row: dict[str, object]) -> bool:
    return _text(row.get("dataset_key")) == BRAIN_STRUCTURE_DATASET_KEY


def _is_healthy_brain_structure(row: dict[str, object]) -> bool:
    return _is_brain_structure(row) and _text(row.get("binary_status")) == "healthy"


def _is_ad_brain_structure(row: dict[str, object]) -> bool:
    return (
        _is_brain_structure(row)
        and _text(row.get("binary_status")) == "unhealthy"
        and _text(row.get("label_family")) == "neurodegenerative"
    )


def _is_tumor_row(row: dict[str, object]) -> bool:
    return _text(row.get("label_family")) == "tumor"


def _project_split(subject_id: str, *, salt: str) -> str:
    digest = hashlib.sha1(f"{salt}:{subject_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") / 2**64
    if bucket < 0.7:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"


def _task_row(
    row: dict[str, object],
    *,
    task_name: str,
    task_label: int | str,
    class_name: str,
    task_split: str,
) -> dict[str, str]:
    task_row = {field: "" for field in TASK_MANIFEST_FIELDNAMES}
    for field in standardized_manifest_fieldnames():
        task_row[field] = _text(row.get(field))
    task_row["task_name"] = task_name
    task_row["task_label"] = _text(task_label)
    task_row["class_name"] = class_name
    task_row["task_split"] = task_split
    return task_row


def _classification_rows(
    rows: list[dict[str, object]],
    *,
    task_name: str,
    selector: Callable[[dict[str, object]], tuple[int, str] | None],
    split_resolver: Callable[[dict[str, object]], str],
) -> list[dict[str, str]]:
    selected_rows: list[dict[str, str]] = []
    for row in rows:
        if _is_excluded(row) or not _has_real_t1(row):
            continue
        selection = selector(row)
        if selection is None:
            continue
        task_label, class_name = selection
        selected_rows.append(
            _task_row(
                row,
                task_name=task_name,
                task_label=task_label,
                class_name=class_name,
                task_split=split_resolver(row),
            )
        )
    return selected_rows


def build_classification_t1_tumor_vs_cn(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Tumor positives plus cognitively normal brain-structure negatives."""

    def selector(row: dict[str, object]) -> tuple[int, str] | None:
        if _is_tumor_row(row):
            return (1, "tumor")
        if _is_healthy_brain_structure(row):
            return (0, "healthy_control")
        return None

    return _classification_rows(
        rows,
        task_name="classification_t1_tumor_vs_cn",
        selector=selector,
        split_resolver=lambda row: _project_split(
            _text(row.get("subject_id")),
            salt="classification_t1_tumor_vs_cn",
        ),
    )


def build_classification_t1_ad_vs_cn(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Alzheimer positives plus cognitively normal controls from brain-structure."""

    def selector(row: dict[str, object]) -> tuple[int, str] | None:
        if _is_ad_brain_structure(row):
            return (1, "alzheimer")
        if _is_healthy_brain_structure(row):
            return (0, "healthy_control")
        return None

    return _classification_rows(
        rows,
        task_name="classification_t1_ad_vs_cn",
        selector=selector,
        split_resolver=lambda row: _text(row.get("source_split"))
        or _project_split(
            _text(row.get("subject_id")),
            salt="classification_t1_ad_vs_cn",
        ),
    )


def build_classification_t1_any_unhealthy_vs_healthy(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Optional broad classification task across all unhealthy sources."""

    def selector(row: dict[str, object]) -> tuple[int, str] | None:
        binary_status = _text(row.get("binary_status"))
        if binary_status == "unhealthy":
            return (1, "unhealthy")
        if binary_status == "healthy":
            return (0, "healthy")
        return None

    return _classification_rows(
        rows,
        task_name="classification_t1_any_unhealthy_vs_healthy",
        selector=selector,
        split_resolver=lambda row: _text(row.get("source_split"))
        or _project_split(
            _text(row.get("subject_id")),
            salt="classification_t1_any_unhealthy_vs_healthy",
        ),
    )


def build_segmentation_binary_curated(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Curated lesion-mask segmentation manifest, excluding brain-structure."""

    return [
        _task_row(
            row,
            task_name="segmentation_binary_curated",
            task_label="",
            class_name="lesion_mask",
            task_split=_text(row.get("source_split"))
            or _project_split(
                _text(row.get("subject_id")),
                salt="segmentation_binary_curated",
            ),
        )
        for row in rows
        if not _is_excluded(row)
        and not _is_brain_structure(row)
        and _text(row.get("task_type")) == "segmentation"
        and _text(row.get("mask_path"))
        and _text(row.get("mask_tier")) == "curated"
    ]


def build_segmentation_binary_broad(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Broader lesion-mask segmentation manifest, excluding brain-structure."""

    return [
        _task_row(
            row,
            task_name="segmentation_binary_broad",
            task_label="",
            class_name="lesion_mask",
            task_split=_text(row.get("source_split"))
            or _project_split(
                _text(row.get("subject_id")),
                salt="segmentation_binary_broad",
            ),
        )
        for row in rows
        if not _is_excluded(row)
        and not _is_brain_structure(row)
        and _text(row.get("task_type")) == "segmentation"
        and _text(row.get("mask_path"))
        and _text(row.get("mask_tier"))
    ]


def build_all_task_manifests(
    rows: list[dict[str, object]],
    *,
    include_any_unhealthy: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Build all requested task manifests from standardized rows."""

    manifests = {
        "classification_t1_tumor_vs_cn.csv": build_classification_t1_tumor_vs_cn(rows),
        "classification_t1_ad_vs_cn.csv": build_classification_t1_ad_vs_cn(rows),
        "segmentation_binary_curated.csv": build_segmentation_binary_curated(rows),
        "segmentation_binary_broad.csv": build_segmentation_binary_broad(rows),
    }
    if include_any_unhealthy:
        manifests["classification_t1_any_unhealthy_vs_healthy.csv"] = (
            build_classification_t1_any_unhealthy_vs_healthy(rows)
        )
    return manifests


def write_task_manifests(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    include_any_unhealthy: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Write task manifest CSVs to output directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifests = build_all_task_manifests(
        rows,
        include_any_unhealthy=include_any_unhealthy,
    )
    for filename, manifest_rows in manifests.items():
        write_csv_rows(destination / filename, manifest_rows, TASK_MANIFEST_FIELDNAMES)
    return manifests
