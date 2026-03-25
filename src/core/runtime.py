"""Runtime helpers for device and reproducibility."""

from __future__ import annotations

import torch
from monai.utils import set_determinism


def resolve_device(prefer: str = "auto") -> torch.device:
    """Resolve device from user preference."""
    normalized = prefer.lower()
    if normalized in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda:0")

    if normalized == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if normalized == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible(seed: int) -> None:
    """Apply deterministic settings for training reproducibility."""
    set_determinism(seed=seed)
