"""Registry for segmentation model backends."""

from __future__ import annotations

from src.segmentation.backends.base import SegmentationBackend
from src.segmentation.backends.monai_unet import MonaiUNetBackend
from src.segmentation.backends.nnunet import NnUNetBackend
from src.segmentation.backends.torch_unet import TorchUNetBackend

_BACKENDS: dict[str, SegmentationBackend] = {
    "monai_unet": MonaiUNetBackend(),
    "torch_unet3d": TorchUNetBackend(),
    "nnunet": NnUNetBackend(),
}


def list_segmentation_backends() -> list[str]:
    """Return all supported segmentation backend names."""
    return sorted(_BACKENDS.keys())


def get_segmentation_backend(name: str) -> SegmentationBackend:
    """Get backend by name."""
    key = name.strip().lower()
    if key not in _BACKENDS:
        available = ", ".join(list_segmentation_backends())
        raise ValueError(f"Unknown segmentation backend '{name}'. Available: {available}")
    return _BACKENDS[key]
