"""Classifier registry for rule-based and learned classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rules import ClassificationThresholds


DEFAULT_CLASSIFIER_REGISTRY_RELATIVE_PATH = Path(
    "res/classification/classifier_registry.json"
)


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
class ClassifierSpec:
    """Definition for one classification pipeline entry."""

    classifier_id: str
    classifier_type: str
    thresholds: ClassificationThresholds


class ClassifierRegistry:
    """Read classifier specs from JSON."""

    def __init__(
        self,
        repo_root: Path | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self.repo_root = resolve_repo_root(repo_root or Path(__file__).resolve())
        self.registry_path = (
            registry_path
            if registry_path is not None
            else self.repo_root / DEFAULT_CLASSIFIER_REGISTRY_RELATIVE_PATH
        )
        if not self.registry_path.is_absolute():
            self.registry_path = self.repo_root / self.registry_path

    def load(self) -> dict[str, ClassifierSpec]:
        if not self.registry_path.is_file():
            raise FileNotFoundError(
                f"Classifier registry not found at: {self.registry_path}"
            )

        with self.registry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        classifiers = payload.get("classifiers")
        if not isinstance(classifiers, list) or not classifiers:
            raise ValueError(
                "Classifier registry is invalid: expected non-empty list at key 'classifiers'"
            )

        resolved: dict[str, ClassifierSpec] = {}
        for item in classifiers:
            spec = self._parse_spec(item)
            if spec.classifier_id in resolved:
                raise ValueError(
                    f"Duplicate classifier id '{spec.classifier_id}' in {self.registry_path}"
                )
            resolved[spec.classifier_id] = spec

        return resolved

    def get(self, classifier_id: str) -> ClassifierSpec:
        specs = self.load()
        if classifier_id not in specs:
            available = ", ".join(sorted(specs.keys()))
            raise KeyError(
                f"Unknown classifier_id '{classifier_id}'. Available classifiers: {available}"
            )
        return specs[classifier_id]

    def _parse_spec(self, raw: Any) -> ClassifierSpec:
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid classifier entry type in {self.registry_path}: {type(raw)!r}"
            )

        classifier_id = str(raw["id"]).strip()
        classifier_type = str(raw.get("type", "rule_based")).strip()
        thresholds_raw = raw.get("thresholds", {})
        if not isinstance(thresholds_raw, dict):
            raise ValueError(
                f"Classifier '{classifier_id}' has invalid thresholds; expected object"
            )

        thresholds = ClassificationThresholds(
            min_tumor_voxels=int(thresholds_raw.get("min_tumor_voxels", 16)),
            enhancing_ratio_for_aggressive=float(
                thresholds_raw.get("enhancing_ratio_for_aggressive", 0.20)
            ),
            core_ratio_for_compact=float(
                thresholds_raw.get("core_ratio_for_compact", 0.70)
            ),
        )

        return ClassifierSpec(
            classifier_id=classifier_id,
            classifier_type=classifier_type,
            thresholds=thresholds,
        )
