"""Classifier-gated segmentation inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.classifier.base import TumorClassifier
from src.inference.service import InferenceService


@dataclass
class PipelineResult:
    """Output object for the gated inference pipeline."""

    has_tumor: bool
    tumor_probability: float
    segmentation_mask: torch.Tensor | None


class TumorSegmentationPipeline:
    """Run classification first and segment only if tumor is likely present."""

    def __init__(
        self,
        classifier: TumorClassifier,
        segmenter,
        classifier_threshold: float = 0.5,
    ) -> None:
        self._service = InferenceService(
            classifier=classifier,
            segmenter=segmenter,
            classifier_threshold=classifier_threshold,
        )

    def run(self, image: torch.Tensor) -> PipelineResult:
        """Run classifier then optionally segmentation on an already loaded image."""
        result = self._service.run_tensor(image)
        return PipelineResult(
            has_tumor=result.has_tumor,
            tumor_probability=result.tumor_probability,
            segmentation_mask=result.segmentation_mask,
        )

    def run_from_sample_path(self, dataset_adapter, sample_path: str) -> PipelineResult:
        """Run pipeline from a dataset sample path using an adapter."""
        result = self._service.run_sample(dataset_adapter, sample_path)
        return PipelineResult(
            has_tumor=result.has_tumor,
            tumor_probability=result.tumor_probability,
            segmentation_mask=result.segmentation_mask,
        )
