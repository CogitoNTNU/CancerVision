"""Model registry helpers for inference-time model discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

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


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if not all(isinstance(key, str) for key in value):
        return False
    return all(torch.is_tensor(tensor) for tensor in value.values())


def _strip_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    prefixed = [key.startswith(prefix) for key in keys]
    if sum(prefixed) < len(keys):
        return state_dict

    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _extract_state_dict(
    checkpoint: Any,
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    if _looks_like_state_dict(checkpoint):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not a mapping and does not contain model weights"
        )

    candidate_keys = (
        "model_state",
        "state_dict",
        "model",
        "model_dict",
        "network",
        "net",
        "weights",
    )
    for key in candidate_keys:
        value = checkpoint.get(key)
        if _looks_like_state_dict(value):
            return value

    for value in checkpoint.values():
        if _looks_like_state_dict(value):
            return value

    raise ValueError(
        f"Checkpoint {checkpoint_path} does not contain a recognizable state_dict "
        "(expected raw state_dict or a key like model_state/state_dict)"
    )


def load_checkpoint_state_dict(
    checkpoint_path: Path,
    *,
    map_location: torch.device,
) -> dict[str, torch.Tensor]:
    """Load and normalize checkpoint content into a plain state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    return _extract_state_dict(checkpoint, checkpoint_path)


def load_weights_with_fallbacks(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    map_location: torch.device,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    """Load model weights from common checkpoint formats and wrapper prefixes."""
    loaded_state_dict = (
        state_dict
        if state_dict is not None
        else load_checkpoint_state_dict(checkpoint_path, map_location=map_location)
    )

    candidate_prefixes = ("", "module.", "model.", "model.module.", "net.", "network.")
    errors: list[str] = []

    for prefix in candidate_prefixes:
        candidate = loaded_state_dict if not prefix else _strip_prefix(loaded_state_dict, prefix)
        try:
            incompatible = model.load_state_dict(candidate, strict=False)
            missing_count = len(incompatible.missing_keys)
            unexpected_count = len(incompatible.unexpected_keys)
            if missing_count == 0 and unexpected_count == 0:
                model.load_state_dict(candidate, strict=True)
                return

            errors.append(
                f"prefix='{prefix or '<none>'}': "
                f"{missing_count} missing keys, {unexpected_count} unexpected keys"
            )
        except RuntimeError as exc:
            error_message = str(exc).split("\n", maxsplit=1)[0]
            errors.append(f"prefix='{prefix or '<none>'}': {error_message}")

    details = " | ".join(errors)
    raise ValueError(
        f"Unable to load checkpoint {checkpoint_path}. Tried key-prefix variants: {details}"
    )
