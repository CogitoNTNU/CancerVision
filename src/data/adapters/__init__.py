"""Dataset adapters."""

from .base import DatasetAdapter
from .brats import BratsAdapter
from .ixi import IxiAdapter

__all__ = ["DatasetAdapter", "BratsAdapter", "IxiAdapter"]
