#!/usr/bin/env python
"""Registry-driven training entrypoint."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Sequence

from .experiment_registry import ExperimentRegistry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model training from a registry-managed experiment id."
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="dynunet_brats_baseline",
        help="Experiment id from res/configs/training_registry.json",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Optional path to a custom training registry JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved trainer command without running it",
    )
    return parser.parse_args(argv)


def _to_flag(name: str) -> str:
    return f"--{name}"


def args_dict_to_argv(arguments: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in arguments.items():
        flag = _to_flag(key)
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{key}")
            continue

        if isinstance(value, list):
            argv.append(flag)
            argv.extend(str(item) for item in value)
            continue

        argv.extend([flag, str(value)])

    return argv


def run_experiment(experiment_id: str, registry_path: str | None, dry_run: bool) -> None:
    registry = ExperimentRegistry(
        repo_root=Path(__file__).resolve(),
        registry_path=Path(registry_path) if registry_path else None,
    )
    spec = registry.get(experiment_id)

    trainer_argv = args_dict_to_argv(spec.arguments)
    print(f"Experiment id  : {spec.experiment_id}", flush=True)
    print(f"Trainer module : {spec.trainer_module}", flush=True)
    print(f"Trainer args   : {' '.join(trainer_argv)}", flush=True)

    if dry_run:
        return

    trainer_module = importlib.import_module(spec.trainer_module)
    train_main = getattr(trainer_module, "main", None)
    if train_main is None:
        raise AttributeError(
            f"Trainer module '{spec.trainer_module}' does not expose a 'main' function"
        )

    train_main(trainer_argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(args.experiment_id, args.registry, args.dry_run)


if __name__ == "__main__":
    main()
