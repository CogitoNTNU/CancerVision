"""IXI adapter implementation for dataset-neutral inference workflows."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import torch

from .base import DatasetAdapter


def _is_nifti_file(path: str) -> bool:
    lower = path.lower()
    return os.path.isfile(path) and (lower.endswith(".nii") or lower.endswith(".nii.gz"))


def _is_ixi_t2_name(filename: str) -> bool:
    lower = filename.lower()
    return lower.startswith("ixi") and (lower.endswith("-t2.nii") or lower.endswith("-t2.nii.gz"))


def _find_ixi_file(path: str) -> str:
    """Resolve a sample path to one concrete IXI NIfTI file."""
    normalized = os.path.abspath(path)

    if os.path.isfile(normalized):
        if not _is_nifti_file(normalized):
            raise FileNotFoundError(f"Expected a NIfTI file (.nii/.nii.gz), got: {path}")
        return normalized

    if not os.path.isdir(normalized):
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Prefer canonical IXI T2 files when selecting from a directory.
    names = sorted(os.listdir(normalized))
    for name in names:
        candidate = os.path.join(normalized, name)
        if _is_nifti_file(candidate) and _is_ixi_t2_name(name):
            return candidate

    # Fallback: any NIfTI file in the folder.
    for name in names:
        candidate = os.path.join(normalized, name)
        if _is_nifti_file(candidate):
            return candidate

    raise FileNotFoundError(f"No NIfTI files found under directory: {path}")


class IxiAdapter(DatasetAdapter):
    """IXI T2 adapter for inference (single modality replicated to 4 channels)."""

    name = "ixi"
    description = "IXI-style T2 NIfTI files (single image per sample)"

    def default_data_dir(self, project_root: str) -> str:
        primary = os.path.join(project_root, "res", "data", "IXI-T2")
        if os.path.isdir(primary):
            return primary

        return os.path.join(project_root, "res", "data", "archive", "IXI-T2")

    def supports_path(self, path: str) -> bool:
        try:
            resolved = _find_ixi_file(path)
        except FileNotFoundError:
            return False

        # Be strict for discovery: only classify as IXI if naming looks like IXI-T2.
        return _is_ixi_t2_name(os.path.basename(resolved))

    def build_training_records(self, data_dir: str) -> list[dict[str, list[str] | str]]:
        raise NotImplementedError(
            "IXI adapter currently supports inference only. "
            "Training records require segmentation labels, which IXI-T2 does not provide in this project."
        )

    def load_inference_image(self, sample_path: str) -> torch.Tensor:
        image_path = _find_ixi_file(sample_path)
        volume = nib.load(image_path).get_fdata(dtype=np.float32)

        if volume.ndim == 4:
            volume = volume[..., 0]
        if volume.ndim != 3:
            raise ValueError(
                "Expected IXI sample volume shape (H, W, D) or (H, W, D, T); "
                f"got {tuple(volume.shape)}"
            )

        # Keep compatibility with existing 4-channel classifier/segmenter checkpoints.
        stacked = np.stack([volume, volume, volume, volume], axis=0)
        return torch.from_numpy(stacked)

    def get_input_channels(self) -> int:
        return 4

    def get_output_channels(self) -> int:
        return 3

    def get_segmentation_label_transform(self):
        return None

    def save_prediction_mask(
        self,
        sample_path: str,
        mask: torch.Tensor,
        output_path: str,
    ) -> None:
        reference_path = _find_ixi_file(sample_path)
        reference_nii = nib.load(reference_path)

        if mask.ndim != 4:
            raise ValueError("Expected mask tensor shape (C, H, W, D)")

        mask_np = mask.cpu().numpy().astype(np.uint8)
        label_map = np.zeros(mask_np.shape[1:], dtype=np.uint8)

        if mask_np.shape[0] >= 3:
            tc = mask_np[0] > 0
            wt = mask_np[1] > 0
            et = mask_np[2] > 0
            label_map[wt] = 1
            label_map[tc] = 2
            label_map[et] = 3
        else:
            label_map = (mask_np[0] > 0).astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pred_nii = nib.Nifti1Image(
            label_map,
            affine=reference_nii.affine,
            header=reference_nii.header,
        )
        nib.save(pred_nii, output_path)
