"""Segmentation inference adapter."""

from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference

from src.core import load_checkpoint, resolve_device
from src.segmentation.registry import get_segmentation_backend


def _normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize common training-time key prefixes for inference-time loading.

    Handles checkpoints saved from:
    - torch.compile models (keys prefixed with ``_orig_mod.``)
    - DataParallel/DistributedDataParallel wrappers (``module.``)
    """
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        k = key
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        normalized[k] = value
    return normalized


class SegmentationInferer:
    """Run segmentation inference for preprocessed 3D volumes."""

    def __init__(
        self,
        model: torch.nn.Module,
        roi_size: tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        device: torch.device | None = None,
        backend_name: str = "monai_unet",
    ) -> None:
        self.device = device or resolve_device("auto")
        self.model = model.to(self.device).eval()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.backend_name = backend_name

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_backend: str | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        roi_size: tuple[int, int, int] = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        device: torch.device | None = None,
    ) -> "SegmentationInferer":
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        metadata = checkpoint.get("metadata", {})

        backend_name = model_backend or metadata.get("model_backend", "monai_unet")
        resolved_in_channels = int(in_channels or metadata.get("in_channels", 4))
        resolved_out_channels = int(out_channels or metadata.get("out_channels", 3))

        backend = get_segmentation_backend(backend_name)
        model = backend.build_model(
            in_channels=resolved_in_channels,
            out_channels=resolved_out_channels,
        )
        state_dict = _normalize_state_dict_keys(checkpoint["model_state_dict"])
        model.load_state_dict(state_dict)
        return cls(
            model=model,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            device=device,
            backend_name=backend_name,
        )

    def predict_mask(self, image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary mask tensor of shape (3, H, W, D)."""
        if image.ndim != 4:
            raise ValueError("Expected image tensor shape (C, H, W, D)")

        image_batch = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = sliding_window_inference(
                image_batch,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self.model,
                overlap=self.overlap,
            )
            probabilities = torch.sigmoid(logits)
            mask = (probabilities >= threshold).float()
        return mask.squeeze(0).cpu()
