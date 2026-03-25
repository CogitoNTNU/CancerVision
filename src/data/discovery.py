"""Dataset discovery helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.data.registry import adapter_items


@dataclass
class DatasetMatch:
    """One discovered dataset candidate."""

    dataset_type: str
    path: str


def discover_datasets(search_dir: str, max_depth: int = 4) -> list[DatasetMatch]:
    """Discover known dataset types under search_dir."""
    root = os.path.abspath(search_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Search directory does not exist: {search_dir}")

    matches: list[DatasetMatch] = []
    visited: set[tuple[str, str]] = set()

    for current_root, dirs, _files in os.walk(root):
        rel = os.path.relpath(current_root, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs[:] = []
            continue

        for dataset_type, adapter in adapter_items():
            if adapter.supports_path(current_root):
                key = (dataset_type, os.path.abspath(current_root))
                if key not in visited:
                    visited.add(key)
                    matches.append(
                        DatasetMatch(dataset_type=dataset_type, path=os.path.abspath(current_root))
                    )

    return matches
