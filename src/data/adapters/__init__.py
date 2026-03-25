"""Dataset adapters."""

from .base import DatasetAdapter
from .brats import BratsAdapter

__all__ = ["DatasetAdapter", "BratsAdapter"]
