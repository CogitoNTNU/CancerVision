"""BraTS adapter implementation for the dataset-neutral data layer."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import torch

from src.datasets import ConvertToMultiChannelBasedOnBratsClassesd

from .base import DatasetAdapter


def _find_nifti(directory: str, pattern: str) -> str:
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(directory, pattern + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {directory}"
    )


def _looks_like_patient_dir(path: str, patient_name: str) -> bool:
    if not (os.path.isdir(path) and patient_name.startswith("BraTS20_Training_")):
        return False

    required = [
        f"{patient_name}_flair",
        f"{patient_name}_t1",
        f"{patient_name}_t1ce",
        f"{patient_name}_t2",
        f"{patient_name}_seg",
    ]
    return all(
        os.path.isfile(os.path.join(path, stem + ".nii"))
        or os.path.isfile(os.path.join(path, stem + ".nii.gz"))
        for stem in required
    )


def _resolve_training_root(path: str) -> str:
    """Resolve a path to the folder that directly contains patient directories."""
    candidates = [
        path,
        os.path.join(path, "MICCAI_BraTS2020_TrainingData"),
        os.path.join(path, "BraTS2020_TrainingData"),
        os.path.join(path, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"),
    ]

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        try:
            children = os.listdir(candidate)
        except OSError:
            continue
        if any(_looks_like_patient_dir(os.path.join(candidate, name), name) for name in children):
            return candidate

    raise FileNotFoundError(
        "Could not locate BraTS2020 training patient folders under path: "
        f"{path}"
    )


class BratsAdapter(DatasetAdapter):
    """BraTS2020 adapter: 4 modalities + segmentation label."""

    name = "brats"
    description = "BraTS-style directory with patient subfolders and NIfTI modalities"

    def default_data_dir(self, project_root: str) -> str:
        new_path = os.path.join(
            project_root,
            "res",
            "datasets",
            "BraTS2020_TrainingData",
            "MICCAI_BraTS2020_TrainingData",
        )
        if os.path.isdir(new_path):
            return new_path

        # Backward compatibility for older repo layout.
        return os.path.join(
            project_root,
            "res",
            "data",
            "dataset",
            "BraTS2020_TrainingData",
            "MICCAI_BraTS2020_TrainingData",
        )

    def supports_path(self, path: str) -> bool:
        try:
            _resolve_training_root(path)
            return True
        except FileNotFoundError:
            return False

    def build_training_records(self, data_dir: str) -> list[dict[str, list[str] | str]]:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        training_root = _resolve_training_root(data_dir)

        records: list[dict[str, list[str] | str]] = []
        for patient_name in sorted(os.listdir(training_root)):
            patient_path = os.path.join(training_root, patient_name)
            if not (
                os.path.isdir(patient_path)
                and patient_name.startswith("BraTS20_Training_")
            ):
                continue

            try:
                flair = _find_nifti(patient_path, f"{patient_name}_flair")
                t1 = _find_nifti(patient_path, f"{patient_name}_t1")
                t1ce = _find_nifti(patient_path, f"{patient_name}_t1ce")
                t2 = _find_nifti(patient_path, f"{patient_name}_t2")
                seg = _find_nifti(patient_path, f"{patient_name}_seg")
            except FileNotFoundError as exc:
                print(f"WARNING: skipping {patient_name} -- {exc}")
                continue

            records.append({"image": [flair, t1, t1ce, t2], "label": seg})

        if not records:
            raise FileNotFoundError(f"No valid BraTS samples found in {training_root}")

        return records

    def load_inference_image(self, sample_path: str) -> torch.Tensor:
        patient_name = os.path.basename(os.path.normpath(sample_path))
        modalities = []
        for suffix in ("flair", "t1", "t1ce", "t2"):
            file_path = _find_nifti(sample_path, f"{patient_name}_{suffix}")
            modalities.append(nib.load(file_path).get_fdata(dtype=np.float32))

        stacked = np.stack(modalities, axis=0)
        return torch.from_numpy(stacked)

    def get_input_channels(self) -> int:
        return 4

    def get_output_channels(self) -> int:
        return 3

    def get_segmentation_label_transform(self):
        return ConvertToMultiChannelBasedOnBratsClassesd(keys="label")

    def save_prediction_mask(
        self,
        sample_path: str,
        mask: torch.Tensor,
        output_path: str,
    ) -> None:
        # Use flair image affine/header as spatial reference for saved prediction.
        patient_name = os.path.basename(os.path.normpath(sample_path))
        flair_path = _find_nifti(sample_path, f"{patient_name}_flair")
        flair_nii = nib.load(flair_path)

        if mask.ndim != 4:
            raise ValueError("Expected mask tensor shape (C, H, W, D)")

        mask_np = mask.cpu().numpy().astype(np.uint8)
        label_map = np.zeros(mask_np.shape[1:], dtype=np.uint8)
        tc = mask_np[0] > 0
        wt = mask_np[1] > 0
        et = mask_np[2] > 0

        # Convert channels back to BraTS-style single-channel labels.
        label_map[wt] = 2
        label_map[tc] = 1
        label_map[et] = 4

        pred_nii = nib.Nifti1Image(label_map, affine=flair_nii.affine, header=flair_nii.header)
        nib.save(pred_nii, output_path)
