"""Segmentation model and backend abstractions."""

from .registry import get_segmentation_backend, list_segmentation_backends

__all__ = ["get_segmentation_backend", "list_segmentation_backends"]
