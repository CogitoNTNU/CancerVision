"""Bridge between the inference `architecture` field and the model registry.

Inference model specs reference architectures by id (e.g. "dynunet_brats_v1").
Those ids map to entries in `src.models.registry`, which is the single source of
truth for how segmentation models are constructed.
"""

from __future__ import annotations

import torch

from src.models.registry import build_model, list_models

from .model_registry import ModelSpec

_ARCHITECTURE_TO_MODEL: dict[str, str] = {
    "dynunet_brats_v1": "dynunet",
    "unet_brats_v1": "unet",
}


def register_architecture(architecture_id: str, model_name: str) -> None:
    """Map a new inference architecture id onto a registered model name."""
    if model_name not in list_models():
        raise KeyError(
            f"Unknown model '{model_name}'. Register it in src.models.registry first."
        )
    _ARCHITECTURE_TO_MODEL[architecture_id.strip()] = model_name


def list_architectures() -> list[str]:
    return sorted(_ARCHITECTURE_TO_MODEL)


def build_model_for_spec(spec: ModelSpec) -> torch.nn.Module:
    """Build a torch model from a model registry spec."""
    model_name = _ARCHITECTURE_TO_MODEL.get(spec.architecture)
    if model_name is None:
        known = ", ".join(sorted(_ARCHITECTURE_TO_MODEL)) or "<none>"
        raise KeyError(
            f"Unknown architecture '{spec.architecture}' for model '{spec.model_id}'. "
            f"Known architecture ids: {known}"
        )

    model = build_model(
        model_name,
        in_channels=spec.in_channels,
        out_channels=spec.out_channels,
    )
    return model
