"""Interfaces for pluggable segmentation model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SegmentationBackend(ABC):
    """Contract for segmentation model implementations and transforms."""

    name: str
    description: str

    @abstractmethod
    def build_model(self, in_channels: int, out_channels: int) -> torch.nn.Module:
        """Build an uninitialized model instance."""

    @abstractmethod
    def get_train_transforms(self, dataset_adapter, roi_size: tuple[int, int, int], num_samples: int):
        """Return MONAI train transforms for this backend and dataset."""

    @abstractmethod
    def get_val_transforms(self, dataset_adapter):
        """Return MONAI validation transforms for this backend and dataset."""
