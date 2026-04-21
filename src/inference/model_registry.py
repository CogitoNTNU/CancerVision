"""Model registry helpers for inference-time model discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_RELATIVE_PATH = Path("res/models/model_registry.json")


def resolve_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by searching for pyproject.toml."""
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find repository root from {current}; missing pyproject.toml"
    )


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a deployable segmentation model."""

    model_id: str
    architecture: str
    checkpoint: Path
    in_channels: int
    out_channels: int
    roi_size: tuple[int, int, int]
    sw_batch_size: int = 4
    overlap: float = 0.5
    input_modalities: tuple[str, ...] = ("flair", "t1", "t1ce", "t2")
    output_labels: tuple[str, ...] = ("tc", "wt", "et")
    threshold: float = 0.5


class ModelRegistry:
    """Read and resolve model specifications from a JSON registry."""

    def __init__(
        self,
        repo_root: Path | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self.repo_root = resolve_repo_root(repo_root or Path(__file__).resolve())
        self.registry_path = (
            registry_path
            if registry_path is not None
            else self.repo_root / DEFAULT_REGISTRY_RELATIVE_PATH
        )
        if not self.registry_path.is_absolute():
            self.registry_path = self.repo_root / self.registry_path

    def load(self) -> dict[str, ModelSpec]:
        if not self.registry_path.is_file():
            raise FileNotFoundError(
                f"Model registry not found at: {self.registry_path}"
            )

        with self.registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        models = payload.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError(
                "Model registry is invalid: expected non-empty list at key 'models'"
            )

        resolved: dict[str, ModelSpec] = {}
        for item in models:
            spec = self._parse_spec(item)
            if spec.model_id in resolved:
                raise ValueError(
                    f"Duplicate model id '{spec.model_id}' in {self.registry_path}"
                )
            resolved[spec.model_id] = spec

        return resolved

    def get(self, model_id: str) -> ModelSpec:
        all_specs = self.load()
        if model_id not in all_specs:
            available = ", ".join(sorted(all_specs.keys()))
            raise KeyError(
                f"Unknown model_id '{model_id}'. Available models: {available}"
            )
        return all_specs[model_id]

    def list_model_ids(self) -> list[str]:
        return sorted(self.load().keys())

    def _parse_spec(self, raw: Any) -> ModelSpec:
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid model entry type in {self.registry_path}: {type(raw)!r}"
            )

        model_id = str(raw["id"]).strip()
        architecture = str(raw["architecture"]).strip()
        checkpoint_raw = str(raw["checkpoint"]).strip()
        checkpoint = self.repo_root / checkpoint_raw

        roi = raw.get("roi_size", [128, 128, 128])
        if not isinstance(roi, list) or len(roi) != 3:
            raise ValueError(
                f"Model '{model_id}' has invalid roi_size: expected list of 3 ints"
            )

        input_modalities = tuple(raw.get("input_modalities", ["flair", "t1", "t1ce", "t2"]))
        output_labels = tuple(raw.get("output_labels", ["tc", "wt", "et"]))

        return ModelSpec(
            model_id=model_id,
            architecture=architecture,
            checkpoint=checkpoint,
            in_channels=int(raw.get("in_channels", 4)),
            out_channels=int(raw.get("out_channels", 3)),
            roi_size=(int(roi[0]), int(roi[1]), int(roi[2])),
            sw_batch_size=int(raw.get("sw_batch_size", 4)),
            overlap=float(raw.get("overlap", 0.5)),
            input_modalities=input_modalities,
            output_labels=output_labels,
            threshold=float(raw.get("threshold", 0.5)),
        )
