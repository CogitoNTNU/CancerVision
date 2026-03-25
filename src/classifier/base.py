"""Common interfaces for tumor classification."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TumorClassifier(ABC):
    """Binary tumor classifier interface."""

    @abstractmethod
    def predict_proba(self, image: torch.Tensor) -> float:
        """Return probability of tumor presence in [0, 1]."""

    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> bool:
        """Return tumor/no-tumor decision from predict_proba."""
        return self.predict_proba(image) >= threshold
