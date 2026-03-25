"""Heuristic classifier used as a fallback when no learned classifier is provided."""

from __future__ import annotations

import torch

from .base import TumorClassifier


class HeuristicTumorClassifier(TumorClassifier):
    """Estimate tumor probability from high-intensity T1ce voxel fraction."""

    def __init__(
        self,
        t1ce_channel: int = 2,
        intensity_threshold: float = 0.8,
        calibration_fraction: float = 0.005,
    ) -> None:
        self.t1ce_channel = t1ce_channel
        self.intensity_threshold = intensity_threshold
        self.calibration_fraction = calibration_fraction

    def predict_proba(self, image: torch.Tensor) -> float:
        if image.ndim != 4:
            raise ValueError("Expected image tensor shape (C, H, W, D)")

        t1ce = image[self.t1ce_channel]
        high_signal_fraction = float((t1ce > self.intensity_threshold).float().mean())
        proba = min(1.0, high_signal_fraction / max(self.calibration_fraction, 1e-6))
        return max(0.0, proba)
