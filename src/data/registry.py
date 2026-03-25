"""Dataset adapter registry."""

from __future__ import annotations

from src.data.adapters.base import DatasetAdapter
from src.data.adapters.brats import BratsAdapter

_ADAPTERS: dict[str, DatasetAdapter] = {
    "brats": BratsAdapter(),
}


def list_dataset_types() -> list[str]:
    """Return available dataset adapter names."""
    return sorted(_ADAPTERS.keys())


def get_dataset_adapter(name: str) -> DatasetAdapter:
    """Fetch adapter by name."""
    key = name.strip().lower()
    if key not in _ADAPTERS:
        available = ", ".join(list_dataset_types())
        raise ValueError(f"Unknown dataset type '{name}'. Available: {available}")
    return _ADAPTERS[key]


def adapter_items() -> list[tuple[str, DatasetAdapter]]:
    """Return adapter name/instance pairs."""
    return sorted(_ADAPTERS.items(), key=lambda x: x[0])
