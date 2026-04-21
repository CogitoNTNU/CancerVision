#!/usr/bin/env python
"""CLI for multi-model BraTS segmentation inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .architectures import build_model_for_spec
from .model_registry import ModelRegistry, resolve_repo_root
from .pipeline import infer_case


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run segmentation inference using a registry of deployable models."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="dynunet_latest",
        help="Model id from res/models/model_registry.json",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Optional path to a custom model registry JSON file",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--case-dir",
        type=str,
        help="Single case directory (expects BraTS naming: <case>_flair/t1/t1ce/t2)",
    )
    input_group.add_argument(
        "--input-root",
        type=str,
        help="Directory containing multiple case folders",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NIfTI file path for --case-dir mode",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output directory for --input-root mode",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override prediction threshold from model registry",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device",
    )
    return parser.parse_args(argv)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda:0")
    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, registry_path: str | None, device: torch.device):
    repo_root = resolve_repo_root(Path(__file__).resolve())
    registry = ModelRegistry(
        repo_root=repo_root,
        registry_path=Path(registry_path) if registry_path is not None else None,
    )
    spec = registry.get(model_id)

    if not spec.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint for model '{model_id}' does not exist: {spec.checkpoint}"
        )

    model = build_model_for_spec(spec)
    checkpoint = torch.load(spec.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state") if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        raise ValueError(
            f"Checkpoint {spec.checkpoint} is missing 'model_state' and is not a raw state_dict"
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, spec


def run_single_case(
    model: torch.nn.Module,
    spec,
    case_dir: Path,
    output: Path | None,
    device: torch.device,
    threshold: float | None,
) -> None:
    out_path = output or (case_dir / f"{case_dir.name}_pred.nii.gz")
    saved = infer_case(
        model=model,
        spec=spec,
        case_dir=case_dir,
        output_path=out_path,
        device=device,
        threshold=threshold,
    )
    print(f"Saved prediction: {saved}", flush=True)


def run_batch(
    model: torch.nn.Module,
    spec,
    input_root: Path,
    output_root: Path,
    device: torch.device,
    threshold: float | None,
) -> None:
    case_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found in {input_root}")

    print(f"Found {len(case_dirs)} cases in {input_root}", flush=True)
    failures = 0
    for case_dir in case_dirs:
        output_path = output_root / f"{case_dir.name}_pred.nii.gz"
        try:
            saved = infer_case(
                model=model,
                spec=spec,
                case_dir=case_dir,
                output_path=output_path,
                device=device,
                threshold=threshold,
            )
            print(f"[{case_dir.name}] saved {saved}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[{case_dir.name}] failed: {exc}", flush=True)

    if failures:
        raise RuntimeError(f"Inference failed for {failures} case(s).")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, spec = load_model(args.model_id, args.registry, device)

    print(f"Model id      : {spec.model_id}", flush=True)
    print(f"Architecture  : {spec.architecture}", flush=True)
    print(f"Checkpoint    : {spec.checkpoint}", flush=True)
    print(f"Device        : {device}", flush=True)

    if args.case_dir:
        run_single_case(
            model=model,
            spec=spec,
            case_dir=Path(args.case_dir).resolve(),
            output=Path(args.output).resolve() if args.output else None,
            device=device,
            threshold=args.threshold,
        )
        return

    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    default_root = resolve_repo_root(Path(__file__).resolve()) / "res" / "predictions" / spec.model_id
    output_root = Path(args.output_root).resolve() if args.output_root else default_root

    run_batch(
        model=model,
        spec=spec,
        input_root=input_root,
        output_root=output_root,
        device=device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

