#!/usr/bin/env python
"""Segmentation training CLI for BraTS.

Pipeline
--------
1. `setup_runtime()` resolves device and distributed launch (single GPU, torchrun, Slurm).
2. `build_brats_data_dicts()` discovers NIfTI patient folders; records are split
   train/val by patient id.
3. `_build_loaders()` builds MONAI DataLoaders. With ``--cache-rate > 0`` the
   deterministic portion of the transform pipeline is cached in RAM
   (`monai.data.CacheDataset`) so only random crops + flips run per step.
4. `build_model(args.model, **kwargs)` constructs the architecture via the
   model registry in `src.models.registry`. Adding a new model is a single
   `register_model` call there; the CLI flag ``--model`` then selects it.
5. Epoch loop: Dice loss (MONAI or fused-CUDA) + Adam + cosine LR + AMP.
   Validation is sharded across all DDP ranks every ``--val-interval`` epochs
   (MONAI DiceMetric all-gathers internally). Rank 0 writes checkpoints and
   logs to W&B.

Key performance flags (see ``--help`` for all):
    --amp-dtype {fp16,bf16}      FP16 + GradScaler (default) or BF16 (A100/H100).
    --cache-rate 1.0             Full in-RAM cache of deterministic transforms.
    --compile                    torch.compile() the model (fixed-shape speedup).
    --memory-format channels_last   NDHWC layout for faster conv on A100.
    --fused-optimizer            torch.optim.Adam(fused=True).
    --fused-dice-loss            Custom CUDA Dice kernel (see src/kernels/).
    --deep-supervision           nnU-Net multi-scale loss on DynUNet.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.multiprocessing as torch_mp
from dotenv import load_dotenv
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, list_data_collate
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

torch_mp.set_sharing_strategy("file_system")

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

MODELS_WITH_DEEP_SUPERVISION = {"dynunet"}


class _NoWandb:
    def log(self, *_a: Any, **_k: Any) -> None: ...
    def finish(self) -> None: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BraTS segmentation model (DynUNet, UNet, ...).",
    )
    parser.add_argument("--model", type=str, default="dynunet", choices=list_models())
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
    parser.add_argument("--train-micro-batch-size", type=int, default=1)
    parser.add_argument("--val-sw-batch-size", type=int, default=1)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Enable CUDA autocast; use --amp-dtype to pick fp16 or bf16.",
    )
    parser.add_argument(
        "--amp-dtype", choices=("fp16", "bf16"), default="fp16",
        help="Autocast dtype. BF16 skips GradScaler (recommended on A100/H100).",
    )
    parser.add_argument(
        "--deep-supervision", action=argparse.BooleanOptionalAction, default=False,
        help="nnU-Net-style deep supervision (DynUNet only).",
    )
    parser.add_argument("--deep-supr-num", type=int, default=2)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False,
        help="If set, deterministic kernels (slower). Otherwise cudnn.benchmark=True.",
    )
    parser.add_argument(
        "--cache-rate", type=float, default=0.0,
        help="Fraction of training data to cache in RAM via MONAI CacheDataset.",
    )
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False,
        help="Apply torch.compile() to the model (benefits from fixed ROI shapes).",
    )
    parser.add_argument(
        "--memory-format", choices=("standard", "channels_last"), default="standard",
        help="Use NDHWC (channels_last_3d) memory format for faster conv on A100.",
    )
    parser.add_argument(
        "--fused-optimizer", action=argparse.BooleanOptionalAction, default=False,
        help="Use torch.optim.AdamW(fused=True) (CUDA fused kernel).",
    )
    parser.add_argument(
        "--fused-dice-loss", action=argparse.BooleanOptionalAction, default=False,
        help="Use the custom CUDA Dice kernel in src/kernels/ instead of MONAI DiceLoss.",
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online",
    )
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args(argv)


def _autocast_dtype(args: argparse.Namespace) -> torch.dtype | None:
    if not args.amp:
        return None
    return torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16


def _autocast(context: RuntimeContext, dtype: torch.dtype | None) -> contextlib.AbstractContextManager:
    if dtype is not None and context.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _scaler_for(dtype: torch.dtype | None, context: RuntimeContext) -> torch.amp.GradScaler | None:
    if dtype is torch.float16 and context.device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def _micro_batch_slices(total: int, micro: int) -> list[slice]:
    if total < 1 or micro < 1:
        raise ValueError("micro-batch sizes must be >= 1")
    size = min(total, micro)
    return [slice(i, min(i + size, total)) for i in range(0, total, size)]


class _CompileHeartbeat:
    """Background thread that prints progress while torch.compile JIT-traces.

    The first forward/backward under ``torch.compile`` can take 30–120 s on
    DynUNet. GPU utilisation is near zero during that window, which looks
    identical to a hang. We print a heartbeat line every ``interval`` seconds
    so the user can tell the process is alive, then stop as soon as the first
    step completes.
    """

    def __init__(self, label: str, interval: float = 10.0) -> None:
        self._label = label
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started: float | None = None

    def __enter__(self) -> "_CompileHeartbeat":
        self._started = time.time()
        print(
            f"[compile] {self._label} — JIT tracing in progress. "
            "GPU util will be ~0% during compile; this is normal.",
            flush=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._started is not None:
            elapsed = time.time() - self._started
            print(f"[compile] {self._label} finished in {elapsed:.1f}s.", flush=True)

    def _run(self) -> None:
        assert self._started is not None
        while not self._stop.wait(self._interval):
            elapsed = time.time() - self._started
            print(f"[compile] still tracing... {elapsed:.0f}s elapsed", flush=True)


def _build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"{args.model}-brats-{job_id}"
    return f"{args.model}-brats-{time.strftime('%Y%m%d-%H%M%S')}"


def _build_loss(args: argparse.Namespace, context: RuntimeContext) -> torch.nn.Module:
    if args.fused_dice_loss:
        if context.device.type != "cuda":
            raise RuntimeError("--fused-dice-loss requires CUDA.")
        from src.kernels import FusedDiceLoss  # lazy import triggers JIT compile

        rank0_print(context, "Loss          : FusedDiceLoss (custom CUDA kernel)")
        return FusedDiceLoss(smooth_nr=0.0, smooth_dr=1e-5)

    rank0_print(context, "Loss          : MONAI DiceLoss(sigmoid, squared_pred)")
    return DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True
    )


def _deep_supervision_loss(
    outputs: torch.Tensor, labels: torch.Tensor, loss_fn: torch.nn.Module
) -> torch.Tensor:
    """Weighted multi-scale Dice loss matching nnU-Net deep supervision.

    DynUNet stacks heads on axis 1 when deep_supervision=True. Weights decay by
    0.5 per level, renormalised to sum to 1; labels are nearest-neighbour
    downsampled to each head's spatial shape.
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
            target = torch.nn.functional.interpolate(labels, size=head.shape[2:], mode="nearest")
        loss = loss + weight * loss_fn(head, target)
    return loss


def _maybe_init_wandb(
    args: argparse.Namespace, context: RuntimeContext, config: dict[str, Any]
) -> Any:
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


def _make_dataset(records, transform, cache_rate: float, cache_workers: int):
    if cache_rate > 0:
        return CacheDataset(
            data=records,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=cache_workers,
            copy_cache=False,
        )
    return Dataset(data=records, transform=transform)


def _build_loaders(
    args: argparse.Namespace, context: RuntimeContext
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    data_dicts = build_brats_data_dicts(args.data_dir)
    rank0_print(context, f"Total patients : {len(data_dicts)}")

    train_dicts, val_dicts = train_test_split(
        data_dicts, test_size=args.test_size, random_state=args.seed
    )
    rank0_print(context, f"Train / Val    : {len(train_dicts)} / {len(val_dicts)}")

    roi_size = tuple(args.roi_size)
    train_ds = _make_dataset(
        train_dicts,
        get_train_transforms(roi_size, args.num_samples),
        args.cache_rate,
        args.cache_num_workers,
    )
    train_sampler = (
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
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )

    val_ds = _make_dataset(
        val_dicts,
        get_val_transforms(),
        args.cache_rate,
        args.cache_num_workers,
    )
    val_sampler = (
        DistributedSampler(
            val_ds,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
        if context.distributed
        else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_sampler


def _cast_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    memory_format: torch.memory_format | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    if memory_format is not None:
        inputs = inputs.to(memory_format=memory_format)
        labels = labels.to(memory_format=memory_format)
    return inputs, labels


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    args: argparse.Namespace,
    context: RuntimeContext,
    sampler: DistributedSampler | None,
    autocast_dtype: torch.dtype | None,
    memory_format: torch.memory_format | None,
) -> tuple[float, float]:
    if sampler is not None:
        sampler.set_epoch(epoch)

    model.train()
    epoch_loss = 0.0
    step_count = 0

    first_step_of_run = epoch == 0 and args.compile
    for step, batch in enumerate(loader, start=1):
        step_start = time.time()
        step_count += 1
        inputs, labels = _cast_inputs(batch, context.device, memory_format)
        slices = _micro_batch_slices(inputs.shape[0], args.train_micro_batch_size)

        compile_banner = (
            _CompileHeartbeat("first training step") if (first_step_of_run and step == 1 and context.is_main)
            else contextlib.nullcontext()
        )

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        with compile_banner:
            for s in slices:
                micro_inputs = inputs[s]
                micro_labels = labels[s]
                weight = micro_inputs.shape[0] / inputs.shape[0]
                with _autocast(context, autocast_dtype):
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
        if context.is_main:
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
    loader: DataLoader,
    post_trans: Compose,
    roi_size: Sequence[int],
    args: argparse.Namespace,
    context: RuntimeContext,
    autocast_dtype: torch.dtype | None,
    memory_format: torch.memory_format | None,
) -> dict[str, float]:
    """All ranks process their shard; MONAI DiceMetric all-gathers for aggregation."""
    predictor = model.module if isinstance(model, DistributedDataParallel) else model
    predictor.eval()
    mean_metric = DiceMetric(include_background=True, reduction="mean")
    batch_metric = DiceMetric(include_background=True, reduction="mean_batch")

    first_val_ever = not getattr(_validate, "_seen_first", False) and args.compile
    _validate._seen_first = True

    with torch.no_grad():
        for val_step, batch in enumerate(loader, start=1):
            inputs, labels = _cast_inputs(batch, context.device, memory_format)
            compile_banner = (
                _CompileHeartbeat("first validation step")
                if (first_val_ever and val_step == 1 and context.is_main)
                else contextlib.nullcontext()
            )
            with compile_banner, _autocast(context, autocast_dtype):
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


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_micro_batch_size < 1 or args.val_sw_batch_size < 1:
        raise ValueError("micro-batch and sliding-window batch sizes must be >= 1")
    if args.deep_supervision and args.model not in MODELS_WITH_DEEP_SUPERVISION:
        raise ValueError(
            f"--deep-supervision is only supported for models: "
            f"{sorted(MODELS_WITH_DEEP_SUPERVISION)}"
        )
    if args.cache_rate < 0 or args.cache_rate > 1:
        raise ValueError("--cache-rate must be in [0, 1]")


def _configure_cudnn(args: argparse.Namespace) -> None:
    if args.deterministic:
        set_determinism(seed=args.seed)
    else:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def _build_model_for_training(
    args: argparse.Namespace, context: RuntimeContext,
    memory_format: torch.memory_format | None,
) -> torch.nn.Module:
    kwargs: dict[str, Any] = {}
    if args.deep_supervision and args.model == "dynunet":
        kwargs.update(deep_supervision=True, deep_supr_num=args.deep_supr_num)
    model = build_model(args.model, **kwargs).to(context.device)
    if memory_format is not None:
        model = model.to(memory_format=memory_format)
    if context.distributed:
        assert context.device_index is not None
        model = DistributedDataParallel(
            model,
            device_ids=[context.device_index],
            output_device=context.device_index,
        )
    if args.compile:
        # Compile AFTER DDP wrapping so compiler sees the wrapped forward.
        # mode="default" is DDP-friendly; reduce-overhead conflicts with bucket resync.
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
    return model


def _build_optimizer(
    args: argparse.Namespace, model: torch.nn.Module, context: RuntimeContext
) -> torch.optim.Optimizer:
    fused = bool(args.fused_optimizer and context.device.type == "cuda")
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=fused,
    )


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    _validate_args(args)

    context = setup_runtime()
    run_name = _build_run_name(args)
    run_dir = Path(args.save_dir).resolve() / run_name
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)

    memory_format = (
        torch.channels_last_3d if args.memory_format == "channels_last" else None
    )
    autocast_dtype = _autocast_dtype(args)

    wandb_run: Any = _NoWandb()
    total_start = time.time()
    try:
        if context.is_main:
            print_config()
        _configure_cudnn(args)

        rank0_print(context, f"Model          : {args.model}")
        rank0_print(context, f"Device         : {context.device}")
        rank0_print(
            context,
            f"Distributed    : {context.distributed} (world_size={context.world_size})",
        )
        rank0_print(context, f"Run directory  : {run_dir}")
        rank0_print(context, f"Data directory : {args.data_dir}")
        rank0_print(context, f"Autocast       : {autocast_dtype}")
        rank0_print(context, f"Memory format  : {args.memory_format}")
        rank0_print(context, f"Compiled model : {args.compile}")
        if args.compile:
            rank0_print(
                context,
                "  torch.compile is ENABLED. The first training step and the first "
                "validation step will each spend 30–120s JIT tracing. Heartbeat "
                "lines will print every 10s during compile.",
            )
        rank0_print(context, f"Fused optimizer: {args.fused_optimizer}")
        rank0_print(context, f"Cache rate     : {args.cache_rate}")

        train_loader, val_loader, sampler = _build_loaders(args, context)
        model = _build_model_for_training(args, context, memory_format)
        loss_fn = _build_loss(args, context)
        optimizer = _build_optimizer(args, model, context)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )
        scaler = _scaler_for(autocast_dtype, context)
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

        wandb_run = _maybe_init_wandb(
            args,
            context,
            {
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
                "amp_dtype": args.amp_dtype,
                "val_sw_batch_size": args.val_sw_batch_size,
                "distributed": context.distributed,
                "world_size": context.world_size,
                "deep_supervision": args.deep_supervision,
                "cache_rate": args.cache_rate,
                "compile": args.compile,
                "memory_format": args.memory_format,
                "fused_optimizer": args.fused_optimizer,
                "fused_dice_loss": args.fused_dice_loss,
            },
        )

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
                autocast_dtype=autocast_dtype,
                memory_format=memory_format,
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
                    autocast_dtype=autocast_dtype,
                    memory_format=memory_format,
                )
                if context.is_main:
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
