"""Dataset and transform helpers for the DynUNet trainer."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Sequence

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from sklearn.model_selection import train_test_split

from src.datasets import (
    BinarizeLabeld,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureFloatLabeld,
    build_brats_data_dicts,
)
from src.datasets.standardize.pathing import resolve_existing_path
from src.models.dynnet_config import DatasetConfig

WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")


def build_data_dicts(data_dir: str) -> list[dict[str, list[str] | str]]:
    return build_brats_data_dicts(data_dir)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _looks_like_windows_drive_path(path_text: str) -> bool:
    return bool(WINDOWS_DRIVE_PATTERN.match(path_text))


def _resolve_manifest_data_path(
    raw_path: str,
    *,
    manifest_dir: Path,
    field_name: str,
    case_id: str,
) -> str:
    candidates: list[str | Path] = []
    if raw_path:
        if Path(raw_path).is_absolute() or _looks_like_windows_drive_path(raw_path):
            candidates.append(raw_path)
        else:
            candidates.append(manifest_dir / raw_path)
            candidates.append(raw_path)

    for candidate in candidates:
        try:
            return os.path.normpath(str(resolve_existing_path(candidate)))
        except FileNotFoundError:
            continue

    joined_candidates = ", ".join(str(candidate) for candidate in candidates) or "<empty>"
    raise FileNotFoundError(
        f"Missing {field_name} for manifest case '{case_id}'. Checked: {joined_candidates}"
    )


def build_cancervision_segmentation_splits(
    task_manifest_path: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest_path = Path(task_manifest_path)
    rows = _read_csv_rows(manifest_path)
    manifest_dir = manifest_path.parent
    allowed_splits = {"train", "val", "test"}
    seen_case_ids: set[str] = set()
    split_rows = {split: [] for split in allowed_splits}

    for index, row in enumerate(rows, start=1):
        if row.get("exclude_reason"):
            continue
        if not row.get("image_path") or not row.get("mask_path"):
            continue

        split_name = (row.get("task_split") or "").strip().lower()
        case_id = (
            row.get("global_case_id")
            or row.get("subject_id")
            or row.get("image_path")
            or f"row_{index}"
        )
        if split_name not in allowed_splits:
            raise RuntimeError(
                f"Unsupported task_split '{row.get('task_split', '')}' for case '{case_id}' "
                f"in manifest {manifest_path}. Expected one of {sorted(allowed_splits)}."
            )
        if case_id in seen_case_ids:
            raise RuntimeError(
                f"Duplicate case '{case_id}' found in CancerVision task manifest: {manifest_path}"
            )
        seen_case_ids.add(case_id)

        split_rows[split_name].append(
            {
                "image": _resolve_manifest_data_path(
                    row["image_path"],
                    manifest_dir=manifest_dir,
                    field_name="image_path",
                    case_id=case_id,
                ),
                "label": _resolve_manifest_data_path(
                    row["mask_path"],
                    manifest_dir=manifest_dir,
                    field_name="mask_path",
                    case_id=case_id,
                ),
            }
        )

    return split_rows["train"], split_rows["val"], split_rows["test"]


def get_brats_train_transforms(roi_size: Sequence[int], num_samples: int) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_brats_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def get_cancervision_binary_seg_train_transforms(
    roi_size: Sequence[int],
    num_samples: int,
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            BinarizeLabeld(keys="label"),
            EnsureFloatLabeld(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_cancervision_binary_seg_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            BinarizeLabeld(keys="label"),
            EnsureFloatLabeld(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def get_dataset_config(dataset_source: str) -> DatasetConfig:
    if dataset_source == "brats":
        return DatasetConfig(
            name="brats",
            in_channels=4,
            out_channels=3,
            metric_names=("tc", "wt", "et"),
            train_transform_builder=get_brats_train_transforms,
            val_transform_builder=get_brats_val_transforms,
        )
    if dataset_source == "cancervision_binary_seg":
        return DatasetConfig(
            name="cancervision_binary_seg",
            in_channels=1,
            out_channels=1,
            metric_names=("lesion",),
            train_transform_builder=get_cancervision_binary_seg_train_transforms,
            val_transform_builder=get_cancervision_binary_seg_val_transforms,
        )
    raise ValueError(f"Unsupported dataset source: {dataset_source}")


def build_dataset_splits(
    args: argparse.Namespace,
) -> tuple[list[dict[str, list[str] | str]], list[dict[str, list[str] | str]], list[dict[str, list[str] | str]]]:
    if args.dataset_source == "brats":
        data_dir = os.path.normpath(args.data_dir)
        data_dicts = build_data_dicts(data_dir)
        train_dicts, val_dicts = train_test_split(
            data_dicts, test_size=args.test_size, random_state=args.seed
        )
        return train_dicts, val_dicts, []

    if args.dataset_source == "cancervision_binary_seg":
        task_manifest = os.path.normpath(args.task_manifest)
        train_dicts, val_dicts, test_dicts = build_cancervision_segmentation_splits(
            task_manifest
        )
        if not train_dicts:
            raise RuntimeError(
                f"No train rows found in CancerVision task manifest: {task_manifest}"
            )
        if not val_dicts:
            raise RuntimeError(
                f"No val rows found in CancerVision task manifest: {task_manifest}"
            )
        return train_dicts, val_dicts, test_dicts

    raise ValueError(f"Unsupported dataset source: {args.dataset_source}")
