#!/usr/bin/env python
"""End-to-end BraTS pipeline entrypoint.

Flow
----
1. `preflight()` validates the environment before anything heavy starts:
     * CUDA availability and device count
     * BraTS data root exists and every patient folder has the 5 expected NIfTI files
     * W&B credentials and mode are usable (unless `--wandb-mode disabled`)
2. `dispatch()` forwards remaining flags into `src.training.train.main()`.

The point of this module is to fail fast on a fresh cloud instance (vast.ai,
Lambda, Slurm worker) rather than crashing mid-epoch with a cryptic error.

Usage
-----
    python -m src.pipeline --model dynunet \
        --data-dir res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
        --max-epochs 100

Forwards every flag accepted by `src.training.train` plus:
    --skip-preflight          do not validate before launching
    --max-missing-patients N  cap the number of malformed patient folders
                              tolerated during preflight (default 0)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from src.datasets.brats import MODALITY_ORDER, PATIENT_DIR_PREFIX
from src.training.train import main as train_main


@dataclass(frozen=True)
class PreflightReport:
    data_dir: Path
    patient_count: int
    missing_patients: list[str]
    cuda_device_count: int
    cuda_device_names: list[str]
    wandb_mode: str
    wandb_has_api_key: bool


def _validate_brats_tree(data_dir: Path) -> tuple[int, list[str]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"BraTS data directory does not exist: {data_dir}")

    required_suffixes = tuple(f"_{m}" for m in MODALITY_ORDER) + ("_seg",)
    patient_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith(PATIENT_DIR_PREFIX)
    )
    if not patient_dirs:
        raise FileNotFoundError(
            f"No BraTS patient folders found under {data_dir} "
            f"(expected names starting with '{PATIENT_DIR_PREFIX}')"
        )

    missing: list[str] = []
    for patient_dir in patient_dirs:
        name = patient_dir.name
        for suffix in required_suffixes:
            found = any(
                (patient_dir / f"{name}{suffix}{ext}").is_file()
                for ext in (".nii", ".nii.gz")
            )
            if not found:
                missing.append(f"{name}{suffix}")
                break
    return len(patient_dirs), missing


def _validate_cuda() -> tuple[int, list[str]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This pipeline is intended for GPU instances "
            "(e.g. vast.ai). Use src.training.train directly if you need CPU."
        )
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("torch reports CUDA available but no visible devices")
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    return count, names


def _validate_wandb(mode: str) -> tuple[str, bool]:
    if mode == "disabled":
        return mode, False

    api_key = os.getenv("WANDB_API_KEY")
    if mode == "online" and not api_key:
        print(
            "WARNING: --wandb-mode=online but WANDB_API_KEY is unset; "
            "falling back to offline logging.",
            flush=True,
        )
        mode = "offline"
    return mode, bool(api_key)


def preflight(
    data_dir: Path,
    wandb_mode: str,
    *,
    max_missing_patients: int = 0,
) -> PreflightReport:
    """Validate environment + data layout. Raises on any hard failure."""
    patient_count, missing = _validate_brats_tree(data_dir)
    if len(missing) > max_missing_patients:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"BraTS tree has {len(missing)} missing files (tolerated "
            f"{max_missing_patients}). Example misses: {sample}"
        )

    cuda_count, cuda_names = _validate_cuda()
    resolved_mode, has_key = _validate_wandb(wandb_mode)

    report = PreflightReport(
        data_dir=data_dir,
        patient_count=patient_count,
        missing_patients=missing,
        cuda_device_count=cuda_count,
        cuda_device_names=cuda_names,
        wandb_mode=resolved_mode,
        wandb_has_api_key=has_key,
    )
    _print_report(report)
    return report


def _print_report(report: PreflightReport) -> None:
    print("=" * 60, flush=True)
    print("Pipeline preflight", flush=True)
    print(f"  Data dir         : {report.data_dir}", flush=True)
    print(f"  Patients         : {report.patient_count}", flush=True)
    if report.missing_patients:
        print(f"  Missing files    : {len(report.missing_patients)} (tolerated)", flush=True)
    print(f"  CUDA devices     : {report.cuda_device_count}", flush=True)
    for i, name in enumerate(report.cuda_device_names):
        print(f"    [{i}] {name}", flush=True)
    print(f"  W&B mode         : {report.wandb_mode}", flush=True)
    print(f"  W&B api key set  : {report.wandb_has_api_key}", flush=True)
    print("=" * 60, flush=True)


def _split_argv(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--max-missing-patients", type=int, default=0)
    known, remaining = parser.parse_known_args(list(argv))
    return known, remaining


def _extract_flag_value(argv: list[str], flag: str, default: str) -> str:
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return default


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    pipeline_args, train_argv = _split_argv(argv)

    data_dir = Path(
        _extract_flag_value(train_argv, "--data-dir", os.environ.get("DATA_DIR", ""))
        or "res/data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    ).resolve()
    wandb_mode = _extract_flag_value(train_argv, "--wandb-mode", "online")

    wants_help = any(token in ("-h", "--help") for token in train_argv)
    if not pipeline_args.skip_preflight and not wants_help:
        preflight(
            data_dir,
            wandb_mode,
            max_missing_patients=pipeline_args.max_missing_patients,
        )

    train_main(train_argv)


if __name__ == "__main__":
    main()
