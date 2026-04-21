"""Checkpoint save/load helpers for segmentation training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel


@dataclass
class ResumeState:
    start_epoch: int = 0
    best_metric: float = -1.0
    best_metric_epoch: int = -1


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    best_metric_epoch: int,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": _unwrap(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_weights(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_unwrap(model).state_dict(), path)


def load_resume(
    resume_path: str | None,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    map_location: torch.device,
) -> ResumeState:
    if not resume_path:
        return ResumeState()

    path = Path(resume_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(path, map_location=map_location)
    target = _unwrap(model)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        target.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        return ResumeState(
            start_epoch=int(checkpoint.get("epoch", -1)) + 1,
            best_metric=float(checkpoint.get("best_metric", -1.0)),
            best_metric_epoch=int(checkpoint.get("best_metric_epoch", -1)),
        )

    target.load_state_dict(checkpoint)
    return ResumeState()
