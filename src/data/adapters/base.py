"""Dataset adapter interface for dataset-neutral pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DatasetAdapter(ABC):
    """Defines how a concrete dataset is discovered and loaded."""

    name: str
    description: str

    @abstractmethod
    def build_training_records(self, data_dir: str) -> list[dict[str, list[str] | str]]:
        """Return records with `image` (paths list) and `label` (path)."""

    @abstractmethod
    def load_inference_image(self, sample_path: str) -> torch.Tensor:
        """Load one sample for inference as a tensor shaped (C, H, W, D)."""

    @abstractmethod
    def get_input_channels(self) -> int:
        """Return expected number of model input channels."""

    @abstractmethod
    def get_output_channels(self) -> int:
        """Return expected number of segmentation output channels."""

    @abstractmethod
    def get_segmentation_label_transform(self):
        """Return a label map transform for segmentation targets, if needed."""

    @abstractmethod
    def save_prediction_mask(
        self,
        sample_path: str,
        mask: torch.Tensor,
        output_path: str,
    ) -> None:
        """Persist predicted mask in dataset-friendly format."""

    @abstractmethod
    def default_data_dir(self, project_root: str) -> str:
        """Return conventional data location for this dataset."""

    @abstractmethod
    def supports_path(self, path: str) -> bool:
        """Check if path likely contains this dataset format."""
