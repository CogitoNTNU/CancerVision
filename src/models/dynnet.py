#!/usr/bin/env python
"""BraTS2020 DynUNet trainer with optional W&B and single-node Slurm DDP."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.utils import set_determinism

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import ConvertToMultiChannelBasedOnBratsClassesd  # noqa: E402

load_dotenv()

DEFAULT_DATA_DIR = REPO_ROOT / "res" / "dataset" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
DEFAULT_SAVE_DIR = REPO_ROOT / "res" / "models"
WANDB_PROJECT = "cancervision"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "cancervision")
DEFAULT_SW_BATCH_SIZE = 4


@dataclass
class RuntimeContext:
    device: torch.device
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


class NoOpWandbRun:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MONAI DynUNet on BraTS2020 NIfTI volumes"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.normpath(str(DEFAULT_DATA_DIR)),
        help="Path to MICCAI_BraTS2020_TrainingData containing patient dirs",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.normpath(str(DEFAULT_SAVE_DIR)),
        help="Directory where run folders and checkpoints are saved",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Training batch size per process"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--val-interval", type=int, default=1, help="Validate every N epochs"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers per process",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of patients reserved for validation",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Random crop ROI size as three integers",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of random crops per volume per iteration",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA AMP during training and validation",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name; defaults to a Slurm-aware generated name",
    )
    return parser.parse_args(argv)


def find_nifti(directory: str, pattern: str) -> str:
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(directory, pattern + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {directory}"
    )


def build_data_dicts(data_dir: str) -> list[dict[str, list[str] | str]]:
    data_dicts: list[dict[str, list[str] | str]] = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    patient_dirs = sorted(
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("BraTS20_Training_")
    )

    for patient_name in patient_dirs:
        patient_path = os.path.join(data_dir, patient_name)
        try:
            flair = find_nifti(patient_path, f"{patient_name}_flair")
            t1 = find_nifti(patient_path, f"{patient_name}_t1")
            t1ce = find_nifti(patient_path, f"{patient_name}_t1ce")
            t2 = find_nifti(patient_path, f"{patient_name}_t2")
            seg = find_nifti(patient_path, f"{patient_name}_seg")
        except FileNotFoundError as exc:
            print(f"WARNING: skipping {patient_name} -- {exc}", flush=True)
            continue

        data_dicts.append({"image": [flair, t1, t1ce, t2], "label": seg})

    if not data_dicts:
        raise FileNotFoundError(f"No valid BraTS patient folders found in {data_dir}")

    return data_dicts


def get_train_transforms(roi_size: Sequence[int], num_samples: int) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def build_model() -> DynUNet:
    return DynUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320],
        dropout=0.2,
        res_block=True,
        deep_supervision=False,
    )


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"dynunet-brats-{job_id}"
    return f"dynunet-brats-{time.strftime('%Y%m%d-%H%M%S')}"


def is_main_process(context: RuntimeContext) -> bool:
    return context.rank == 0


def should_use_amp(args: argparse.Namespace, context: RuntimeContext) -> bool:
    return bool(args.amp and context.device.type == "cuda")


def setup_device_and_distributed() -> RuntimeContext:
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA for this trainer.")
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", local_rank)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return RuntimeContext(
        device=device,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed(context: RuntimeContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def synchronize(context: RuntimeContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.barrier()


def rank0_print(context: RuntimeContext, message: str) -> None:
    if is_main_process(context):
        print(message, flush=True)


def maybe_init_wandb(
    args: argparse.Namespace,
    context: RuntimeContext,
    config: dict[str, Any],
) -> Any:
    if not is_main_process(context) or args.wandb_mode == "disabled":
        return NoOpWandbRun()

    try:
        import wandb
    except Exception:
        print("W&B unavailable in this environment; continuing without it.", flush=True)
        return NoOpWandbRun()

    mode = args.wandb_mode
    api_key = os.getenv("WANDB_API_KEY")
    if mode == "online" and not api_key:
        print("WANDB_API_KEY missing; falling back to offline mode.", flush=True)
        mode = "offline"

    try:
        if mode == "online" and api_key:
            wandb.login(key=api_key, relogin=True)
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=config,
            name=config["run_name"],
            mode=mode,
        )
        return run if run is not None else NoOpWandbRun()
    except Exception as exc:
        print(f"W&B init failed ({exc}); continuing without it.", flush=True)
        return NoOpWandbRun()


def get_autocast_context(context: RuntimeContext, enabled: bool) -> contextlib.AbstractContextManager:
    if enabled and context.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def reduce_mean(value: float, count: int, context: RuntimeContext) -> float:
    if not context.distributed:
        return value / max(count, 1)
    reduced = torch.tensor([value, float(count)], device=context.device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return (reduced[0] / reduced[1].clamp_min(1.0)).item()


def load_resume_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    resume_path: str | None,
    context: RuntimeContext,
) -> tuple[int, float, int]:
    if not resume_path:
        return 0, -1.0, -1

    path = Path(resume_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(path, map_location=context.device)
    model_to_load = model.module if isinstance(model, DistributedDataParallel) else model

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_to_load.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
        best_metric_epoch = int(checkpoint.get("best_metric_epoch", -1))
        return start_epoch, best_metric, best_metric_epoch

    model_to_load.load_state_dict(checkpoint)
    return 0, -1.0, -1


def save_last_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    best_metric_epoch: int,
) -> None:
    model_state = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
    }
    torch.save(checkpoint, run_dir / "last_checkpoint.pt")


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_function: DiceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    context: RuntimeContext,
    train_sampler: DistributedSampler | None,
) -> tuple[float, float]:
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    epoch_loss = 0.0
    step_count = 0

    for step, batch_data in enumerate(train_loader, start=1):
        step_start = time.time()
        step_count += 1
        inputs = batch_data["image"].to(context.device, non_blocking=True)
        labels = batch_data["label"].to(context.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(context, should_use_amp(args, context)):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        if is_main_process(context) and step % 10 == 0:
            print(
                f"  step {step}/{len(train_loader)}"
                f"  train_loss: {loss.item():.4f}"
                f"  step_time: {time.time() - step_start:.2f}s",
                flush=True,
            )

    scheduler.step()
    mean_epoch_loss = reduce_mean(epoch_loss, step_count, context)
    current_lr = scheduler.get_last_lr()[0]
    return mean_epoch_loss, current_lr


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader | None,
    post_trans: Compose,
    roi_size: Sequence[int],
    args: argparse.Namespace,
    context: RuntimeContext,
) -> dict[str, float] | None:
    if not is_main_process(context) or val_loader is None:
        return None

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    predictor = model.module if isinstance(model, DistributedDataParallel) else model
    predictor.eval()

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(context.device, non_blocking=True)
            val_labels = val_data["label"].to(context.device, non_blocking=True)
            with get_autocast_context(context, should_use_amp(args, context)):
                val_outputs = sliding_window_inference(
                    val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=DEFAULT_SW_BATCH_SIZE,
                    predictor=predictor,
                    overlap=0.5,
                )
            val_outputs_list = [post_trans(item) for item in decollate_batch(val_outputs)]
            val_labels_list = decollate_batch(val_labels)
            dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            dice_metric_batch(y_pred=val_outputs_list, y=val_labels_list)

    metric = dice_metric.aggregate().item()
    metric_batch = dice_metric_batch.aggregate()
    dice_metric.reset()
    dice_metric_batch.reset()

    return {
        "dice_mean": metric,
        "dice_tc": metric_batch[0].item(),
        "dice_wt": metric_batch[1].item(),
        "dice_et": metric_batch[2].item(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = setup_device_and_distributed()

    run_name = build_run_name(args)
    run_dir = Path(args.save_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True) if is_main_process(context) else None
    synchronize(context)

    wandb_run: Any = NoOpWandbRun()
    total_start = time.time()
    try:
        if is_main_process(context):
            print_config()
        set_determinism(seed=args.seed)

        rank0_print(context, f"Using device: {context.device}")
        rank0_print(context, f"Distributed: {context.distributed} (world_size={context.world_size})")
        rank0_print(context, f"Run directory: {run_dir}")

        data_dir = os.path.normpath(args.data_dir)
        rank0_print(context, f"Data directory : {data_dir}")
        rank0_print(context, f"Exists         : {os.path.isdir(data_dir)}")

        data_dicts = build_data_dicts(data_dir)
        rank0_print(context, f"Total patients : {len(data_dicts)}")
        train_dicts, val_dicts = train_test_split(
            data_dicts, test_size=args.test_size, random_state=args.seed
        )
        rank0_print(context, f"Train patients : {len(train_dicts)}")
        rank0_print(context, f"Val patients   : {len(val_dicts)}")

        roi_size = tuple(args.roi_size)
        train_ds = Dataset(
            data=train_dicts,
            transform=get_train_transforms(roi_size, args.num_samples),
        )
        train_sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
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

        val_loader: DataLoader | None = None
        if is_main_process(context):
            val_ds = Dataset(data=val_dicts, transform=get_val_transforms())
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=pin_memory,
            )

        model = build_model().to(context.device)
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )

        loss_function = DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )
        scaler = (
            torch.cuda.amp.GradScaler(enabled=True)
            if should_use_amp(args, context)
            else None
        )
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        start_epoch, best_metric, best_metric_epoch = load_resume_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resume_path=args.resume,
            context=context,
        )
        if args.resume:
            rank0_print(
                context,
                f"Resumed from {args.resume} starting at epoch {start_epoch + 1}",
            )

        wandb_config = {
            "run_name": run_name,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_interval": args.val_interval,
            "seed": args.seed,
            "roi_size": list(roi_size),
            "num_samples": args.num_samples,
            "test_size": args.test_size,
            "amp": args.amp,
            "distributed": context.distributed,
            "world_size": context.world_size,
        }
        wandb_run = maybe_init_wandb(args, context, wandb_config)

        for epoch in range(start_epoch, args.max_epochs):
            epoch_start = time.time()
            rank0_print(context, "-" * 40)
            rank0_print(context, f"Epoch {epoch + 1}/{args.max_epochs}")

            avg_loss, current_lr = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                context=context,
                train_sampler=train_sampler,
            )

            if is_main_process(context):
                print(f"  avg loss: {avg_loss:.4f}  lr: {current_lr:.2e}", flush=True)
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "epoch": epoch + 1,
                    }
                )

            synchronize(context)
            if (epoch + 1) % args.val_interval == 0:
                metrics = validate(
                    model=model,
                    val_loader=val_loader,
                    post_trans=post_trans,
                    roi_size=roi_size,
                    args=args,
                    context=context,
                )
                if is_main_process(context) and metrics is not None:
                    if metrics["dice_mean"] > best_metric:
                        best_metric = metrics["dice_mean"]
                        best_metric_epoch = epoch + 1
                        model_state = (
                            model.module.state_dict()
                            if isinstance(model, DistributedDataParallel)
                            else model.state_dict()
                        )
                        torch.save(model_state, run_dir / "best_metric_model.pth")
                        print(
                            f"  -> saved new best model to {run_dir / 'best_metric_model.pth'}",
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

            if is_main_process(context):
                save_last_checkpoint(
                    run_dir=run_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    best_metric_epoch=best_metric_epoch,
                )
                print(f"  epoch time: {time.time() - epoch_start:.1f}s", flush=True)

            synchronize(context)

        if is_main_process(context):
            total_time = time.time() - total_start
            print("=" * 60, flush=True)
            print("Training complete", flush=True)
            print(
                f"  Best mean Dice : {best_metric:.4f} @ epoch {best_metric_epoch}",
                flush=True,
            )
            print(f"  Total time     : {total_time:.1f}s ({total_time / 3600:.2f}h)", flush=True)
            print(f"  Checkpoint     : {run_dir / 'best_metric_model.pth'}", flush=True)
            print("=" * 60, flush=True)
    finally:
        wandb_run.finish()
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
