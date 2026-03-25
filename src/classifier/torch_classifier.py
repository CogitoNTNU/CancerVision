"""Torch-based binary tumor classifier."""

from __future__ import annotations

import torch
from torch import nn

from src.core import load_checkpoint

from .base import TumorClassifier


class SmallTumorClassifier3D(nn.Module):
    """Compact 3D CNN for tumor presence classification."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(output_size=1),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features).squeeze(1)


class TorchTumorClassifier(TumorClassifier):
    """Adapter exposing a torch module through the TumorClassifier interface."""

    def __init__(self, model: nn.Module, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

    def predict_proba(self, image: torch.Tensor) -> float:
        if image.ndim != 4:
            raise ValueError("Expected image tensor shape (C, H, W, D)")

        with torch.no_grad():
            logits = self.model(image.unsqueeze(0).to(self.device))
            proba = torch.sigmoid(logits)[0].item()
        return float(proba)


def load_torch_classifier(
    checkpoint_path: str, device: torch.device | None = None
) -> TorchTumorClassifier:
    """Load a TorchTumorClassifier from a checkpoint file."""
    model = SmallTumorClassifier3D(in_channels=4)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return TorchTumorClassifier(model=model, device=device)
