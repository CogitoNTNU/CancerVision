"""Registry loader for training experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRAINING_REGISTRY_RELATIVE_PATH = Path("res/configs/training_registry.json")


def resolve_repo_root(start: Path | None = None) -> Path:
    """Resolve repository root by locating pyproject.toml."""
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
class TrainingSpec:
    """Configuration for one trainable experiment."""

    experiment_id: str
    trainer_module: str
    arguments: dict[str, Any]


class ExperimentRegistry:
    """Read experiment configurations from JSON registry."""

    def __init__(
        self,
        repo_root: Path | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self.repo_root = resolve_repo_root(repo_root or Path(__file__).resolve())
        self.registry_path = (
            registry_path
            if registry_path is not None
            else self.repo_root / DEFAULT_TRAINING_REGISTRY_RELATIVE_PATH
        )
        if not self.registry_path.is_absolute():
            self.registry_path = self.repo_root / self.registry_path

    def load(self) -> dict[str, TrainingSpec]:
        if not self.registry_path.is_file():
            raise FileNotFoundError(
                f"Training registry not found at: {self.registry_path}"
            )

        with self.registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        experiments = payload.get("experiments")
        if not isinstance(experiments, list) or not experiments:
            raise ValueError(
                "Training registry is invalid: expected non-empty list at key 'experiments'"
            )

        resolved: dict[str, TrainingSpec] = {}
        for item in experiments:
            spec = self._parse_spec(item)
            if spec.experiment_id in resolved:
                raise ValueError(
                    f"Duplicate experiment id '{spec.experiment_id}' in {self.registry_path}"
                )
            resolved[spec.experiment_id] = spec

        return resolved

    def get(self, experiment_id: str) -> TrainingSpec:
        specs = self.load()
        if experiment_id not in specs:
            available = ", ".join(sorted(specs.keys()))
            raise KeyError(
                f"Unknown experiment_id '{experiment_id}'. Available experiments: {available}"
            )
        return specs[experiment_id]

    def list_experiment_ids(self) -> list[str]:
        return sorted(self.load().keys())

    def _parse_spec(self, raw: Any) -> TrainingSpec:
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid experiment entry type in {self.registry_path}: {type(raw)!r}"
            )

        experiment_id = str(raw["id"]).strip()
        trainer_module = str(raw["trainer_module"]).strip()
        arguments = raw.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(
                f"Experiment '{experiment_id}' has invalid arguments; expected object"
            )

        return TrainingSpec(
            experiment_id=experiment_id,
            trainer_module=trainer_module,
            arguments=arguments,
        )
