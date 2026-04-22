#!/usr/bin/env python
"""End-to-end BraTS pipeline entrypoint.

Responsibilities:
  1. Preflight — validate CUDA, BraTS tree integrity, W&B credentials,
     optionally compile and sanity-check the fused Dice CUDA kernel.
  2. Dispatch — forward remaining flags into `src.training.train.main()`.

The point is to fail fast on a fresh cloud instance (vast.ai, Lambda, Slurm)
rather than crash an hour into a run.

Usage:
    python -m src.pipeline [--preflight-only] [--skip-preflight]
                           [--max-missing-patients N]
                           <training-flags...>
    python -m src.pipeline --help
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
from src.training.train import parse_args as parse_training_args


@dataclass(frozen=True)
class PreflightReport:
    data_dir: Path
    patient_count: int
    missing_patients: list[str]
    cuda_device_count: int
    cuda_device_names: list[str]
    wandb_mode: str
    wandb_has_api_key: bool
    fused_dice_loss: bool


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
            if not any(
                (patient_dir / f"{name}{suffix}{ext}").is_file()
                for ext in (".nii", ".nii.gz")
            ):
                missing.append(f"{name}{suffix}")
                break
    return len(patient_dirs), missing


def _validate_cuda() -> tuple[int, list[str]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This pipeline is intended for GPU instances. "
            "Use src.training.train directly for CPU/MPS fallback."
        )
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("torch reports CUDA available but no visible devices")
    return count, [torch.cuda.get_device_name(i) for i in range(count)]


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


def _validate_fused_dice_kernel() -> None:
    """Compile the custom CUDA Dice kernel and verify it matches MONAI."""
    import torch

    from monai.losses import DiceLoss as MonaiDiceLoss

    from src.kernels import FusedDiceLoss

    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16, 16, device="cuda", requires_grad=True)
    targets = (torch.rand(2, 3, 16, 16, 16, device="cuda") > 0.5).float()

    fused = FusedDiceLoss(smooth_nr=0.0, smooth_dr=1e-5)
    reference = MonaiDiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True, reduction="mean"
    )

    loss_fused = fused(logits, targets)
    loss_ref = reference(logits, targets)
    diff = (loss_fused - loss_ref).abs().item()
    if diff > 1e-4:
        raise RuntimeError(
            f"FusedDiceLoss disagrees with MONAI DiceLoss (|Δ|={diff:.2e}). "
            "Refusing to run training with a miscompiled kernel."
        )

    # Gradient check
    loss_fused.backward()
    grad_fused = logits.grad.detach().clone()
    logits.grad = None
    loss_ref.backward()
    grad_ref = logits.grad.detach()
    grad_diff = (grad_fused - grad_ref).abs().max().item()
    if grad_diff > 1e-4:
        raise RuntimeError(
            f"FusedDiceLoss gradient disagrees with MONAI (max|Δ|={grad_diff:.2e})."
        )


def preflight(
    data_dir: Path,
    wandb_mode: str,
    *,
    max_missing_patients: int = 0,
    fused_dice_loss: bool = False,
) -> PreflightReport:
    patient_count, missing = _validate_brats_tree(data_dir)
    if len(missing) > max_missing_patients:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"BraTS tree has {len(missing)} missing files (tolerated "
            f"{max_missing_patients}). Example misses: {sample}"
        )

    cuda_count, cuda_names = _validate_cuda()
    resolved_mode, has_key = _validate_wandb(wandb_mode)

    if fused_dice_loss:
        print("Compiling + verifying FusedDiceLoss CUDA kernel...", flush=True)
        _validate_fused_dice_kernel()
        print("  FusedDiceLoss matches MONAI DiceLoss within tolerance.", flush=True)

    report = PreflightReport(
        data_dir=data_dir,
        patient_count=patient_count,
        missing_patients=missing,
        cuda_device_count=cuda_count,
        cuda_device_names=cuda_names,
        wandb_mode=resolved_mode,
        wandb_has_api_key=has_key,
        fused_dice_loss=fused_dice_loss,
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
    print(f"  Fused Dice loss  : {report.fused_dice_loss}", flush=True)
    print("=" * 60, flush=True)


def _split_pipeline_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split argv into (pipeline_flags, forwarded_training_flags).

    We intentionally parse the training flags with the real training parser
    (imported from src.training.train) so `python -m src.pipeline --help`
    shows the full combined CLI rather than two disjoint surfaces.
    """
    pipeline_parser = argparse.ArgumentParser(add_help=False)
    pipeline_parser.add_argument("--skip-preflight", action="store_true")
    pipeline_parser.add_argument("--preflight-only", action="store_true")
    pipeline_parser.add_argument("--max-missing-patients", type=int, default=0)
    pipeline_args, remaining = pipeline_parser.parse_known_args(argv)
    return pipeline_args, remaining


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)

    # --help should print the full combined CLI without forcing a preflight run.
    if any(token in ("-h", "--help") for token in argv):
        print(
            "Pipeline-level flags:\n"
            "  --skip-preflight          skip all preflight checks\n"
            "  --preflight-only          run preflight then exit (no training)\n"
            "  --max-missing-patients N  tolerate N BraTS files missing (default 0)\n\n"
            "Training flags follow:\n",
            flush=True,
        )
        parse_training_args(["--help"])  # argparse exits

    pipeline_args, train_argv = _split_pipeline_flags(argv)
    # Parse training flags eagerly so typos fail before preflight runs.
    training_args = parse_training_args(train_argv)

    if not pipeline_args.skip_preflight:
        preflight(
            Path(training_args.data_dir).resolve(),
            training_args.wandb_mode,
            max_missing_patients=pipeline_args.max_missing_patients,
            fused_dice_loss=training_args.fused_dice_loss,
        )

    if pipeline_args.preflight_only:
        print("Preflight OK; exiting due to --preflight-only.", flush=True)
        return

    train_main(train_argv)


if __name__ == "__main__":
    main()
