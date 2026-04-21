"""Architecture builders used by inference model specs."""

from __future__ import annotations

from collections.abc import Callable

import torch

from src.models.dynnet import build_model as build_dynunet_model

from .model_registry import ModelSpec

ModelBuilder = Callable[[], torch.nn.Module]


_ARCHITECTURE_BUILDERS: dict[str, ModelBuilder] = {
    "dynunet_brats_v1": build_dynunet_model,
}


def register_architecture(name: str, builder: ModelBuilder) -> None:
    """Register a model builder for a new architecture id."""
    normalized = name.strip()
    if not normalized:
        raise ValueError("Architecture name cannot be empty")
    _ARCHITECTURE_BUILDERS[normalized] = builder


def build_model_for_spec(spec: ModelSpec) -> torch.nn.Module:
    """Build a torch model from a model registry spec."""
    builder = _ARCHITECTURE_BUILDERS.get(spec.architecture)
    if builder is None:
        available = ", ".join(sorted(_ARCHITECTURE_BUILDERS.keys()))
        raise KeyError(
            f"Unknown architecture '{spec.architecture}' for model '{spec.model_id}'. "
            f"Registered architectures: {available}"
        )

    model = builder()

    if hasattr(model, "in_channels") and int(model.in_channels) != spec.in_channels:
        raise ValueError(
            f"Model '{spec.model_id}' expects in_channels={spec.in_channels}, "
            f"builder returned in_channels={int(model.in_channels)}"
        )

    if hasattr(model, "out_channels") and int(model.out_channels) != spec.out_channels:
        raise ValueError(
            f"Model '{spec.model_id}' expects out_channels={spec.out_channels}, "
            f"builder returned out_channels={int(model.out_channels)}"
        )

    return model
