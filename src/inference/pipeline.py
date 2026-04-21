"""Reusable utilities for case-level segmentation inference."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from .model_registry import ModelSpec

MODALITY_SUFFIXES: dict[str, str] = {
    "flair": "_flair",
    "t1": "_t1",
    "t1ce": "_t1ce",
    "t2": "_t2",
}


def _normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    normalized = volume.astype(np.float32, copy=True)
    nonzero = normalized != 0
    if not np.any(nonzero):
        return normalized

    values = normalized[nonzero]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        normalized[nonzero] = values - mean
        return normalized

    normalized[nonzero] = (values - mean) / std
    return normalized


def find_modality_file(case_dir: Path, case_id: str, modality: str) -> Path:
    if modality not in MODALITY_SUFFIXES:
        known = ", ".join(sorted(MODALITY_SUFFIXES.keys()))
        raise KeyError(f"Unknown modality '{modality}'. Supported modalities: {known}")

    suffix = MODALITY_SUFFIXES[modality]
    for extension in (".nii.gz", ".nii"):
        candidate = case_dir / f"{case_id}{suffix}{extension}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find modality '{modality}' for case '{case_id}' in {case_dir}"
    )


def load_case_modalities(
    case_dir: Path,
    modality_order: tuple[str, ...],
) -> tuple[torch.Tensor, nib.Nifti1Image]:
    case_id = case_dir.name
    files = [find_modality_file(case_dir, case_id, modality) for modality in modality_order]

    reference = nib.load(str(files[0]))
    stacked = []
    for file_path in files:
        image = nib.load(str(file_path))
        volume = np.asarray(image.get_fdata(dtype=np.float32))
        stacked.append(_normalize_nonzero(volume))

    image_np = np.stack(stacked, axis=0)
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
    return image_tensor, reference


def channels_to_brats_labels(probability_channels: torch.Tensor, threshold: float) -> np.ndarray:
    """Convert predicted channels (TC, WT, ET) to a single-label BraTS mask."""
    if probability_channels.ndim != 4 or probability_channels.shape[0] != 3:
        raise ValueError(
            "Expected channel-first tensor with shape (3, H, W, D) for TC, WT, ET"
        )

    tc = probability_channels[0] > threshold
    wt = probability_channels[1] > threshold
    et = probability_channels[2] > threshold

    label_map = torch.zeros_like(tc, dtype=torch.uint8)
    label_map[wt] = 2
    label_map[tc] = 1
    label_map[et] = 4
    return label_map.cpu().numpy()


@torch.no_grad()
def infer_case(
    model: torch.nn.Module,
    spec: ModelSpec,
    case_dir: Path,
    output_path: Path,
    device: torch.device,
    threshold: float | None = None,
) -> Path:
    """Run segmentation inference for one case directory and save output mask."""
    inputs, reference = load_case_modalities(case_dir, spec.input_modalities)
    inputs = inputs.to(device, non_blocking=device.type == "cuda")

    logits = sliding_window_inference(
        inputs,
        roi_size=spec.roi_size,
        sw_batch_size=spec.sw_batch_size,
        predictor=model,
        overlap=spec.overlap,
    )
    probabilities = torch.sigmoid(logits[0]).cpu()
    label_map = channels_to_brats_labels(
        probabilities,
        threshold=spec.threshold if threshold is None else threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_image = nib.Nifti1Image(label_map.astype(np.uint8), reference.affine)
    nib.save(prediction_image, str(output_path))
    return output_path
