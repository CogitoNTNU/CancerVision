#!/usr/bin/env python
"""Segmentation training CLI for BraTS.

High-level flow
---------------
1. `setup_runtime()` resolves device + distributed launch (single GPU, torchrun, Slurm).
2. `build_brats_data_dicts()` scans the BraTS NIfTI tree and returns patient records.
3. Records are split into train/val; MONAI Datasets + DataLoaders are built with the
   training/validation transforms from `src.datasets.brats`.
4. `build_model(args.model)` constructs the architecture via the model registry in
   `src.models.registry`. To add a new model, register a builder there; the CLI
   flag `--model <name>` then selects it. Only `dynunet` is wired end-to-end for now.
5. Standard epoch loop: Dice loss + Adam + cosine LR, AMP on CUDA, sliding-window
   validation every `--val-interval` epochs on the main process. Best mean Dice and
   a full last-epoch checkpoint are written under `<save-dir>/<run-name>/`.
6. Weights & Biases logging is optional; W&B init is rank-0 only and falls back
   cleanly when the library or API key is unavailable.

Usage
-----
    python -m src.training.train --model dynunet --data-dir <path-to-brats-root>

Run `python -m src.training.train --help` for the full flag list.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from dotenv import load_dotenv
from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from src.datasets.brats import (
    build_brats_data_dicts,
    get_train_transforms,
    get_val_transforms,
)
from src.models.registry import build_model, list_models
from src.training.checkpoint import load_resume, save_checkpoint, save_weights
from src.training.distributed import (
    RuntimeContext,
    barrier,
    cleanup,
    rank0_print,
    reduce_mean,
    setup_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "res"
    / "data"
    / "brats"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
)
DEFAULT_SAVE_DIR = REPO_ROOT / "res" / "models"
WANDB_PROJECT = "cancervision"


class _NoWandb:
    def log(self, *_a: Any, **_k: Any) -> None: ...
    def finish(self) -> None: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BraTS segmentation model (DynUNet, UNet, ...)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dynunet",
        choices=list_models(),
        help="Registered model name from src.models.registry",
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--train-micro-batch-size",
        type=int,
        default=1,
        help="Sub-batch size used inside each optimizer step to cap 3D memory.",
    )
    parser.add_argument(
        "--val-sw-batch-size",
        type=int,
        default=1,
        help="Sliding-window batch size during validation inference.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA autocast + GradScaler (ignored off-CUDA).",
    )
    parser.add_argument(
        "--deep-supervision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train with nnU-Net-style deep supervision (DynUNet only).",
    )
    parser.add_argument(
        "--deep-supr-num",
        type=int,
        default=2,
        help="Number of extra supervision heads when --deep-supervision is set.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, force deterministic cuDNN kernels (slower). Otherwise enables "
            "cudnn.benchmark for fastest convolutions on fixed input shapes."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args(argv)


def _use_amp(args: argparse.Namespace, context: RuntimeContext) -> bool:
    return bool(args.amp and context.device.type == "cuda")


def _autocast(context: RuntimeContext, enabled: bool) -> contextlib.AbstractContextManager:
    if enabled and context.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _deep_supervision_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: DiceLoss,
) -> torch.Tensor:
    """Weighted multi-scale Dice loss matching nnU-Net's deep-supervision recipe.

    DynUNet stacks supervision heads on axis 1 when `deep_supervision=True`.
    Weights decay by 0.5 per level and are renormalised so they sum to 1;
    labels are trilinearly downsampled to match each head's spatial shape.
    """
    num_heads = outputs.shape[1]
    weights = [0.5 ** i for i in range(num_heads)]
    total = sum(weights)
    weights = [w / total for w in weights]

    loss = outputs.new_zeros(())
    for i, weight in enumerate(weights):
        head = outputs[:, i]
        if head.shape[2:] == labels.shape[2:]:
            target = labels
        else:
            target = torch.nn.functional.interpolate(
                labels, size=head.shape[2:], mode="nearest"
            )
        loss = loss + weight * loss_fn(head, target)
    return loss


def _micro_batch_slices(total: int, micro: int) -> list[slice]:
    if total < 1 or micro < 1:
        raise ValueError("micro-batch sizes must be >= 1")
    size = min(total, micro)
    return [slice(i, min(i + size, total)) for i in range(0, total, size)]


def _build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{args.model}-brats-{job_id}"
    return f"{args.model}-brats-{time.strftime('%Y%m%d-%H%M%S')}"


def _maybe_init_wandb(args: argparse.Namespace, context: RuntimeContext, config: dict[str, Any]) -> Any:
    if not context.is_main or args.wandb_mode == "disabled":
        return _NoWandb()
    try:
        import wandb
    except Exception:
        print("W&B unavailable; continuing without it.", flush=True)
        return _NoWandb()

    mode = args.wandb_mode
    api_key = os.getenv("WANDB_API_KEY")
    if mode == "online" and not api_key:
        print("WANDB_API_KEY missing; falling back to offline.", flush=True)
        mode = "offline"

    try:
        if mode == "online" and api_key:
            wandb.login(key=api_key, relogin=True)
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=os.getenv("WANDB_ENTITY", "cancervision"),
            config=config,
            name=config["run_name"],
            mode=mode,
        )
        return run or _NoWandb()
    except Exception as exc:
        print(f"W&B init failed ({exc}); continuing without it.", flush=True)
        return _NoWandb()


def _build_loaders(
    args: argparse.Namespace, context: RuntimeContext
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    data_dicts = build_brats_data_dicts(args.data_dir)
    rank0_print(context, f"Total patients : {len(data_dicts)}")

    train_dicts, val_dicts = train_test_split(
        data_dicts, test_size=args.test_size, random_state=args.seed
    )
    rank0_print(context, f"Train / Val    : {len(train_dicts)} / {len(val_dicts)}")

    roi_size = tuple(args.roi_size)
    train_ds = Dataset(
        data=train_dicts,
        transform=get_train_transforms(roi_size, args.num_samples),
    )
    sampler = (
        DistributedSampler(
            train_ds, num_replicas=context.world_size, rank=context.rank, shuffle=True
        )
        if context.distributed
        else None
    )

    num_workers = min(args.num_workers, os.cpu_count() or 1)
    pin_memory = context.device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )

    val_loader: DataLoader | None = None
    if context.is_main:
        val_ds = Dataset(data=val_dicts, transform=get_val_transforms())
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, sampler


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: DiceLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    args: argparse.Namespace,
    context: RuntimeContext,
    sampler: DistributedSampler | None,
) -> tuple[float, float]:
    if sampler is not None:
        sampler.set_epoch(epoch)

    model.train()
    epoch_loss = 0.0
    step_count = 0

    for step, batch in enumerate(loader, start=1):
        step_start = time.time()
        step_count += 1
        inputs = batch["image"].to(context.device, non_blocking=True)
        labels = batch["label"].to(context.device, non_blocking=True)
        slices = _micro_batch_slices(inputs.shape[0], args.train_micro_batch_size)

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for s in slices:
            micro_inputs = inputs[s]
            micro_labels = labels[s]
            weight = micro_inputs.shape[0] / inputs.shape[0]
            with _autocast(context, _use_amp(args, context)):
                outputs = model(micro_inputs)
                if args.deep_supervision and outputs.dim() == 6:
                    loss = _deep_supervision_loss(outputs, micro_labels, loss_fn)
                else:
                    loss = loss_fn(outputs, micro_labels)
                scaled = loss * weight
            step_loss += loss.item() * weight
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        epoch_loss += step_loss
        if context.is_main and step % 10 == 0:
            print(
                f"  step {step}/{len(loader)}  train_loss: {step_loss:.4f}"
                f"  step_time: {time.time() - step_start:.2f}s",
                flush=True,
            )

    scheduler.step()
    return reduce_mean(epoch_loss, step_count, context), scheduler.get_last_lr()[0]


def _validate(
    *,
    model: torch.nn.Module,
    loader: DataLoader | None,
    post_trans: Compose,
    roi_size: Sequence[int],
    args: argparse.Namespace,
    context: RuntimeContext,
) -> dict[str, float] | None:
    if not context.is_main or loader is None:
        return None

    predictor = model.module if isinstance(model, DistributedDataParallel) else model
    predictor.eval()
    mean_metric = DiceMetric(include_background=True, reduction="mean")
    batch_metric = DiceMetric(include_background=True, reduction="mean_batch")

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(context.device, non_blocking=True)
            labels = batch["label"].to(context.device, non_blocking=True)
            with _autocast(context, _use_amp(args, context)):
                logits = sliding_window_inference(
                    inputs,
                    roi_size=roi_size,
                    sw_batch_size=args.val_sw_batch_size,
                    predictor=predictor,
                    overlap=0.5,
                )
            preds = [post_trans(p) for p in decollate_batch(logits)]
            gts = decollate_batch(labels)
            mean_metric(y_pred=preds, y=gts)
            batch_metric(y_pred=preds, y=gts)

    mean = mean_metric.aggregate().item()
    per_class = batch_metric.aggregate()
    mean_metric.reset()
    batch_metric.reset()
    return {
        "dice_mean": mean,
        "dice_tc": per_class[0].item(),
        "dice_wt": per_class[1].item(),
        "dice_et": per_class[2].item(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    if args.train_micro_batch_size < 1 or args.val_sw_batch_size < 1:
        raise ValueError("micro-batch and sliding-window batch sizes must be >= 1")

    context = setup_runtime()
    run_name = _build_run_name(args)
    run_dir = Path(args.save_dir).resolve() / run_name
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)

    wandb_run: Any = _NoWandb()
    total_start = time.time()
    try:
        if context.is_main:
            print_config()
        if args.deterministic:
            set_determinism(seed=args.seed)
        else:
            torch.manual_seed(args.seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        rank0_print(context, f"Model          : {args.model}")
        rank0_print(context, f"Device         : {context.device}")
        rank0_print(
            context,
            f"Distributed    : {context.distributed} (world_size={context.world_size})",
        )
        rank0_print(context, f"Run directory  : {run_dir}")
        rank0_print(context, f"Data directory : {args.data_dir}")

        train_loader, val_loader, sampler = _build_loaders(args, context)

        model_kwargs: dict[str, Any] = {}
        if args.model == "dynunet" and args.deep_supervision:
            model_kwargs = {
                "deep_supervision": True,
                "deep_supr_num": args.deep_supr_num,
            }
        elif args.deep_supervision:
            rank0_print(
                context,
                f"WARNING: --deep-supervision requested but model '{args.model}' "
                "does not support it; ignoring.",
            )
            args.deep_supervision = False
        model = build_model(args.model, **model_kwargs).to(context.device)
        if context.distributed:
            assert context.device_index is not None
            model = DistributedDataParallel(
                model,
                device_ids=[context.device_index],
                output_device=context.device_index,
            )

        loss_fn = DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )
        scaler = (
            torch.cuda.amp.GradScaler(enabled=True) if _use_amp(args, context) else None
        )
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        resume = load_resume(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=context.device,
        )
        if args.resume:
            rank0_print(context, f"Resumed from {args.resume} @ epoch {resume.start_epoch + 1}")

        wandb_config = {
            "run_name": run_name,
            "model": args.model,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_interval": args.val_interval,
            "seed": args.seed,
            "roi_size": list(args.roi_size),
            "num_samples": args.num_samples,
            "train_micro_batch_size": args.train_micro_batch_size,
            "test_size": args.test_size,
            "amp": args.amp,
            "val_sw_batch_size": args.val_sw_batch_size,
            "distributed": context.distributed,
            "world_size": context.world_size,
        }
        wandb_run = _maybe_init_wandb(args, context, wandb_config)

        best_metric = resume.best_metric
        best_metric_epoch = resume.best_metric_epoch
        roi_size = tuple(args.roi_size)

        for epoch in range(resume.start_epoch, args.max_epochs):
            epoch_start = time.time()
            rank0_print(context, "-" * 40)
            rank0_print(context, f"Epoch {epoch + 1}/{args.max_epochs}")

            avg_loss, current_lr = _train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                args=args,
                context=context,
                sampler=sampler,
            )

            if context.is_main:
                print(f"  avg loss: {avg_loss:.4f}  lr: {current_lr:.2e}", flush=True)
                wandb_run.log(
                    {"train/loss": avg_loss, "train/lr": current_lr, "epoch": epoch + 1}
                )

            barrier(context)
            if (epoch + 1) % args.val_interval == 0:
                metrics = _validate(
                    model=model,
                    loader=val_loader,
                    post_trans=post_trans,
                    roi_size=roi_size,
                    args=args,
                    context=context,
                )
                if context.is_main and metrics is not None:
                    if metrics["dice_mean"] > best_metric:
                        best_metric = metrics["dice_mean"]
                        best_metric_epoch = epoch + 1
                        save_weights(run_dir / "best_metric_model.pth", model)
                        print(
                            f"  -> new best model saved to {run_dir / 'best_metric_model.pth'}",
                            flush=True,
                        )
                    wandb_run.log(
                        {
                            "val/dice_mean": metrics["dice_mean"],
                            "val/dice_tc": metrics["dice_tc"],
                            "val/dice_wt": metrics["dice_wt"],
                            "val/dice_et": metrics["dice_et"],
                            "val/best_dice": best_metric,
                            "epoch": epoch + 1,
                        }
                    )
                    print(
                        f"  val dice: {metrics['dice_mean']:.4f}"
                        f"  (TC={metrics['dice_tc']:.4f}"
                        f"  WT={metrics['dice_wt']:.4f}"
                        f"  ET={metrics['dice_et']:.4f})"
                        f"\n  best dice: {best_metric:.4f} @ epoch {best_metric_epoch}",
                        flush=True,
                    )

            if context.is_main:
                save_checkpoint(
                    run_dir / "last_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_metric_epoch=best_metric_epoch,
                )
                print(f"  epoch time: {time.time() - epoch_start:.1f}s", flush=True)
            barrier(context)

        if context.is_main:
            total = time.time() - total_start
            print("=" * 60, flush=True)
            print("Training complete", flush=True)
            print(f"  Best mean Dice : {best_metric:.4f} @ epoch {best_metric_epoch}", flush=True)
            print(f"  Total time     : {total:.1f}s ({total / 3600:.2f}h)", flush=True)
            print(f"  Checkpoint     : {run_dir / 'best_metric_model.pth'}", flush=True)
            print("=" * 60, flush=True)
    finally:
        wandb_run.finish()
        cleanup(context)


if __name__ == "__main__":
    main()
