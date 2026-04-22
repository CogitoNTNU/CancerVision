"""BraTS2020 NIfTI dataset: discovery, transforms, label conversion.

The expected on-disk layout is the MICCAI BraTS 2020 training release:

    <root>/BraTS20_Training_001/
        BraTS20_Training_001_flair.nii(.gz)
        BraTS20_Training_001_t1.nii(.gz)
        BraTS20_Training_001_t1ce.nii(.gz)
        BraTS20_Training_001_t2.nii(.gz)
        BraTS20_Training_001_seg.nii(.gz)

Modalities are stacked in a fixed order: (flair, t1, t1ce, t2).
Segmentation labels use the BraTS convention: 0=bg, 1=necrotic, 2=edema, 4=enhancing.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Sequence

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

MODALITY_ORDER: tuple[str, ...] = ("flair", "t1", "t1ce", "t2")
PATIENT_DIR_PREFIX = "BraTS20_Training_"


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """Convert a BraTS label map (0/1/2/4) to a 3-channel binary target.

    Channels:
        TC (Tumor Core)     = {1, 4}
        WT (Whole Tumor)    = {1, 2, 4}
        ET (Enhancing Tumor)= {4}
    """

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key][0]
            tc = torch.logical_or(label == 1, label == 4)
            wt = torch.logical_or(tc, label == 2)
            et = label == 4
            d[key] = torch.stack([tc, wt, et], dim=0).float()
        return d


def _find_nifti(directory: str, pattern: str) -> str:
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(directory, pattern + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {directory}"
    )


def build_brats_data_dicts(data_dir: str | Path) -> list[dict[str, list[str] | str]]:
    """Scan a BraTS training root and return MONAI-style data dicts."""
    data_dir = str(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    patient_dirs = sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
        and name.startswith(PATIENT_DIR_PREFIX)
    )

    data_dicts: list[dict[str, list[str] | str]] = []
    for patient_name in patient_dirs:
        patient_path = os.path.join(data_dir, patient_name)
        try:
            modalities = [
                _find_nifti(patient_path, f"{patient_name}_{m}") for m in MODALITY_ORDER
            ]
            seg = _find_nifti(patient_path, f"{patient_name}_seg")
        except FileNotFoundError as exc:
            print(f"WARNING: skipping {patient_name} -- {exc}", flush=True)
            continue
        data_dicts.append({"image": modalities, "label": seg})

    if not data_dicts:
        raise FileNotFoundError(f"No valid BraTS patient folders found in {data_dir}")
    return data_dicts


def get_train_transforms(roi_size: Sequence[int], num_samples: int) -> Compose:
    """Training pipeline with nnU-Net-style augmentations for BraTS."""
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
            RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys="image",
                prob=0.15,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            ),
            RandScaleIntensityd(keys="image", factors=0.25, prob=0.3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
            RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),
        ]
    )


def get_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
