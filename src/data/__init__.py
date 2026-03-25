"""Dataset-neutral data access layer."""

from .discovery import DatasetMatch, discover_datasets
from .registry import get_dataset_adapter, list_dataset_types

__all__ = ["DatasetMatch", "discover_datasets", "get_dataset_adapter", "list_dataset_types"]
