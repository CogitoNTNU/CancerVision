"""nnUNet backend placeholder.

This module establishes integration points for a future nnUNet trainer.
"""

from __future__ import annotations

from .base import SegmentationBackend


class NnUNetBackend(SegmentationBackend):
    """Placeholder backend to reserve CLI and architecture integration."""

    name = "nnunet"
    description = "Reserved backend for nnUNet integration"

    def build_model(self, in_channels: int, out_channels: int):
        raise NotImplementedError(
            "nnUNet backend is not implemented yet. Add nnUNet integration in "
            "src/segmentation/backends/nnunet.py."
        )

    def get_train_transforms(self, dataset_adapter, roi_size: tuple[int, int, int], num_samples: int):
        raise NotImplementedError("nnUNet backend transforms are not implemented.")

    def get_val_transforms(self, dataset_adapter):
        raise NotImplementedError("nnUNet backend transforms are not implemented.")
