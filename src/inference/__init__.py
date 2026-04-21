"""Inference utilities and CLI entrypoints for segmentation models."""

from .architectures import build_model_for_spec, list_architectures, register_architecture
from .model_registry import ModelRegistry, ModelSpec, resolve_repo_root
from .pipeline import channels_to_brats_labels, infer_case

__all__ = [
    "ModelRegistry",
    "ModelSpec",
    "build_model_for_spec",
    "list_architectures",
    "register_architecture",
    "resolve_repo_root",
    "infer_case",
    "channels_to_brats_labels",
]
