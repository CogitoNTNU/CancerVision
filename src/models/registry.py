"""Segmentation-model registry.

Adding a new model is a two-step process:

    1. Implement a builder function that returns a torch.nn.Module with the
       expected BraTS signature (4 input channels, 3 output channels).
    2. Register it by name via @register_model("my_model") or by calling
       register_model("my_model", builder).

The training CLI selects the model via `--model <name>`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from monai.networks.nets import UNet

from .dynunet import build_dynunet

ModelBuilder = Callable[..., torch.nn.Module]

_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder | None = None):
    """Register a model builder. Usable as a decorator or direct call."""
    normalized = name.strip()
    if not normalized:
        raise ValueError("Model name cannot be empty")

    def _register(fn: ModelBuilder) -> ModelBuilder:
        if normalized in _REGISTRY:
            raise ValueError(f"Model '{normalized}' is already registered")
        _REGISTRY[normalized] = fn
        return fn

    if builder is not None:
        _register(builder)
        return builder
    return _register


def build_model(name: str, **kwargs) -> torch.nn.Module:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY)


register_model("dynunet", build_dynunet)
register_model(
    "unet",
    lambda in_channels=4, out_channels=3, dropout=0.2: UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=dropout,
    ),
)
