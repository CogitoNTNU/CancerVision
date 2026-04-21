"""Runtime helpers for ad-hoc inference driven by the web interface."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src.inference.architectures import build_model_for_spec
from src.inference.model_registry import ModelSpec
from src.inference.pipeline import _normalize_nonzero, channels_to_brats_labels


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda:0")
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_from_weights(
    architecture: str,
    checkpoint_path: Path,
    device: torch.device,
    *,
    in_channels: int = 4,
    out_channels: int = 3,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    threshold: float = 0.5,
) -> tuple[torch.nn.Module, ModelSpec]:
    """Build a model from an arbitrary checkpoint file on disk."""
    spec = ModelSpec(
        model_id="user_upload",
        architecture=architecture,
        checkpoint=checkpoint_path,
        in_channels=in_channels,
        out_channels=out_channels,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        threshold=threshold,
    )

    model = build_model_for_spec(spec)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing 'model_state' and is not a raw state_dict"
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, spec


def _load_and_stack(modality_files: list[Path]) -> tuple[torch.Tensor, nib.Nifti1Image]:
    reference = nib.load(str(modality_files[0]))
    stacked = []
    for path in modality_files:
        image = nib.load(str(path))
        volume = np.asarray(image.get_fdata(dtype=np.float32))
        stacked.append(_normalize_nonzero(volume))

    image_np = np.stack(stacked, axis=0)
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
    return image_tensor, reference


@torch.no_grad()
def infer_from_files(
    model: torch.nn.Module,
    spec: ModelSpec,
    modality_files: list[Path],
    output_path: Path,
    device: torch.device,
    threshold: float | None = None,
) -> tuple[Path, dict[str, int]]:
    """Run inference on a pre-ordered list of modality files and save the mask."""
    if len(modality_files) != len(spec.input_modalities):
        raise ValueError(
            f"Expected {len(spec.input_modalities)} modality files "
            f"({', '.join(spec.input_modalities)}), got {len(modality_files)}"
        )

    inputs, reference = _load_and_stack(modality_files)
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

    label_counts = {
        "background_voxels": int(np.sum(label_map == 0)),
        "tc_voxels": int(np.sum(label_map == 1)),
        "wt_voxels": int(np.sum(label_map == 2)),
        "et_voxels": int(np.sum(label_map == 4)),
    }
    return output_path, label_counts
