"""Higher-level inference service with classifier gating and dataset-aware exports."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.classifier.base import TumorClassifier
from src.data.preprocess import preprocess_image_volume
from src.inference.segmenter import SegmentationInferer


@dataclass
class InferenceResult:
    """Inference output with optional segmentation mask."""

    has_tumor: bool
    tumor_probability: float
    segmentation_mask: torch.Tensor | None


class InferenceService:
    """Coordinates classifier-gated segmentation with dataset adapters."""

    def __init__(
        self,
        classifier: TumorClassifier,
        segmenter: SegmentationInferer,
        classifier_threshold: float = 0.5,
    ) -> None:
        self.classifier = classifier
        self.segmenter = segmenter
        self.classifier_threshold = classifier_threshold

    def run_tensor(self, image: torch.Tensor) -> InferenceResult:
        """Run inference for one preloaded tensor image (C, H, W, D)."""
        normalized_image = preprocess_image_volume(image)
        tumor_probability = self.classifier.predict_proba(normalized_image)
        has_tumor = tumor_probability >= self.classifier_threshold

        if not has_tumor:
            return InferenceResult(
                has_tumor=False,
                tumor_probability=tumor_probability,
                segmentation_mask=None,
            )

        mask = self.segmenter.predict_mask(normalized_image)
        return InferenceResult(
            has_tumor=True,
            tumor_probability=tumor_probability,
            segmentation_mask=mask,
        )

    def run_sample(self, dataset_adapter, sample_path: str) -> InferenceResult:
        """Run inference for one dataset sample loaded through an adapter."""
        image = dataset_adapter.load_inference_image(sample_path)
        return self.run_tensor(image)

    def save_prediction(self, dataset_adapter, sample_path: str, result: InferenceResult, output_path: str) -> None:
        """Save segmentation prediction with adapter-aware format conversion."""
        if result.segmentation_mask is None:
            raise ValueError("No segmentation mask available: classifier gated inference off.")
        dataset_adapter.save_prediction_mask(sample_path, result.segmentation_mask, output_path)
