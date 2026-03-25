"""Checkpoint serialization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    metadata: dict[str, Any],
) -> None:
    """Save model state and metadata in one checkpoint file."""
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            **metadata,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint and normalize legacy structures."""
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint.setdefault("metadata", {})
        return checkpoint

    # Backward compatibility: raw state_dict only.
    return {
        "model_state_dict": checkpoint,
        "metadata": {},
    }
