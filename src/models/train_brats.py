#!/usr/bin/env python
"""Compatibility wrapper for the DynUNet BraTS trainer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import dynnet  # noqa: E402

from src.datasets import ConvertToMultiChannelBasedOnBratsClassesd  # noqa: E402

# ---------------------------------------------------------------------------
# Load config from .env if available
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DistributedContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_env_int(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    """Read an integer environment variable with a default fallback."""
    env = os.environ if env is None else env
    value = env.get(name)
    if value in (None, ""):
        return default
    return int(value)


def bootstrap_distributed_env(env: Mapping[str, str] | None = None):
    """Populate torch.distributed env vars from Slurm when torchrun is not used."""
    if env is not None:
        target_env = env
    else:
        target_env = os.environ

    if "SLURM_PROCID" not in target_env:
        return

    target_env.setdefault("RANK", str(get_env_int("SLURM_PROCID", 0, target_env)))
    target_env.setdefault("LOCAL_RANK", str(get_env_int("SLURM_LOCALID", 0, target_env)))
    target_env.setdefault("WORLD_SIZE", str(get_env_int("SLURM_NTASKS", 1, target_env)))

    if "MASTER_ADDR" not in target_env:
        nodelist = target_env.get("SLURM_STEP_NODELIST") or target_env.get("SLURM_NODELIST")
        if nodelist:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", nodelist],
                    text=True,
                ).splitlines()
            except (OSError, subprocess.CalledProcessError):
                hostnames = []
            if hostnames:
                target_env["MASTER_ADDR"] = hostnames[0]
        target_env.setdefault("MASTER_ADDR", "127.0.0.1")

    if "MASTER_PORT" not in target_env:
        job_id = get_env_int("SLURM_JOB_ID", 29500, target_env)
        target_env["MASTER_PORT"] = str(10000 + (job_id % 50000))


def get_distributed_env(env: Mapping[str, str] | None = None) -> tuple[int, int, int]:
    """Resolve rank metadata from torchrun or Slurm environments."""
    bootstrap_distributed_env(env)
    env = os.environ if env is None else env
    world_size = get_env_int("WORLD_SIZE", -1, env)
    rank = get_env_int("RANK", -1, env)
    local_rank = get_env_int("LOCAL_RANK", -1, env)

    if world_size < 0:
        world_size = get_env_int("SLURM_NTASKS", 1, env)
    if rank < 0:
        rank = get_env_int("SLURM_PROCID", 0, env)
    if local_rank < 0:
        local_rank = get_env_int("SLURM_LOCALID", 0, env)

    return rank, local_rank, world_size


def resolve_cuda_device_index(local_rank: int, visible_device_count: int) -> int:
    """Map a process local rank onto a visible CUDA device index."""
    if visible_device_count < 1:
        raise RuntimeError("CUDA is available but no visible devices were reported")
    if visible_device_count == 1:
        return 0
    if local_rank < 0 or local_rank >= visible_device_count:
        raise RuntimeError(
            f"Invalid local rank {local_rank} for {visible_device_count} visible CUDA devices"
        )
    return local_rank


def setup_distributed_context() -> DistributedContext:
    """Initialize distributed training when launched with multiple processes."""
    rank, local_rank, world_size = get_distributed_env()
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        device_index = resolve_cuda_device_index(local_rank, torch.cuda.device_count())
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        backend = "nccl"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if is_distributed:
            raise RuntimeError("Distributed multi-process training is not supported on MPS")
        device = torch.device("mps")
        backend = None
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return DistributedContext(
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed():
    """Tear down the process group if distributed training was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext):
    """Synchronize distributed workers at epoch boundaries."""
    if not context.is_distributed:
        return
    if context.device.type == "cuda":
        dist.barrier(device_ids=[context.device.index])
        return
    dist.barrier()


def reduce_sum(value: float, context: DistributedContext) -> float:
    """Sum a scalar across all processes."""
    if not context.is_distributed:
        return float(value)

    tensor = torch.tensor(value, device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def log_main(context: DistributedContext, *args, **kwargs):
    """Print only from the main process to avoid duplicate logs."""
    if context.is_main_process:
        print(*args, **kwargs)


def init_wandb(args, context: DistributedContext):
    """Initialize a single W&B run from rank 0 only."""
    if not context.is_main_process:
        return None

    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_entity = os.getenv("WANDB_ENTITY", "cancervision")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    return wandb.init(
        project="cancervision",
        entity=wandb_entity,
        config={
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_interval": args.val_interval,
            "seed": args.seed,
            "roi_size": args.roi_size,
            "num_samples": args.num_samples,
            "test_size": args.test_size,
            "world_size": context.world_size,
        },
    )


def find_nifti(directory: str, pattern: str) -> str:
    """Find a NIfTI file matching *pattern* inside *directory*.

    Checks for both .nii and .nii.gz extensions and returns the first match.
    """
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(directory, pattern + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {directory}"
    )


def build_data_dicts(data_dir: str):
    """Scan BraTS 2020/2023/2024 patient folders and return data dicts.

    Each dict has keys: "image" (list of 4 modality paths) and "label" (seg path).
    """
    return dynnet.build_data_dicts(data_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Train 3D U-Net on BraTS 2020/2023/2024 NIfTI volumes"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.normpath(str(dynnet.DEFAULT_DATA_DIR)),
        help="Path to BraTS 2020/2023/2024 root containing patient dirs",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.normpath(
            os.path.join(script_dir, "..", "..", "res", "models")
        ),
        help="Directory to save the best model checkpoint",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-process training batch size (1 for 3D)",
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
        help="DataLoader workers (0 = main process)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of patients for validation",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Random crop ROI size (3 ints, e.g. 128 128 128)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of random crops per volume per iteration",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transforms(roi_size, num_samples):
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


def get_val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    context = setup_distributed_context()
    wandb_run = None

    try:
        if context.is_main_process:
            print_config()

        log_main(
            context,
            f"Using device: {context.device} "
            f"(rank={context.rank}, local_rank={context.local_rank}, world_size={context.world_size})",
        )

        wandb_run = init_wandb(args, context)

        # Reproducibility
        set_determinism(seed=args.seed)

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        data_dir = os.path.normpath(args.data_dir)
        log_main(context, f"Data directory : {data_dir}")
        log_main(context, f"Exists         : {os.path.isdir(data_dir)}")

        data_dicts = build_data_dicts(data_dir)
        log_main(context, f"Total patients : {len(data_dicts)}")

        train_dicts, val_dicts = train_test_split(
            data_dicts, test_size=args.test_size, random_state=42
        )
        log_main(context, f"Train patients : {len(train_dicts)}")
        log_main(context, f"Val patients   : {len(val_dicts)}")

        roi_size = tuple(args.roi_size)
        train_transform = get_train_transforms(roi_size, args.num_samples)
        val_transform = get_val_transforms()

        train_ds = Dataset(data=train_dicts, transform=train_transform)
        val_ds = Dataset(data=val_dicts, transform=val_transform)

        train_sampler = None
        if context.is_distributed:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
            )

        num_workers = min(args.num_workers, os.cpu_count() or 1)
        log_main(context, f"DataLoader num_workers: {num_workers}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        # ------------------------------------------------------------------
        # Model / loss / optimizer
        # ------------------------------------------------------------------
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.2,
        ).to(context.device)

        if context.is_distributed:
            ddp_kwargs = {}
            if context.device.type == "cuda":
                ddp_kwargs = {
                    "device_ids": [context.device.index],
                    "output_device": context.device.index,
                }
            model = DistributedDataParallel(model, **ddp_kwargs)

        model_for_inference = (
            model.module if isinstance(model, DistributedDataParallel) else model
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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # ------------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------------
        save_dir = os.path.normpath(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)

        best_metric = -1.0
        best_metric_epoch = -1
        metric_values_tc = []
        metric_values_wt = []
        metric_values_et = []

        total_start = time.time()
        for epoch in range(args.max_epochs):
            epoch_start = time.time()
            log_main(context, "-" * 40)
            log_main(context, f"Epoch {epoch + 1}/{args.max_epochs}")

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            epoch_loss_sum = 0.0
            step_count = 0
            for batch_data in train_loader:
                step_start = time.time()
                step_count += 1
                inputs = batch_data["image"].to(context.device)
                labels = batch_data["label"].to(context.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                if context.is_main_process and step_count % 10 == 0:
                    print(
                        f"  step {step_count}/{len(train_loader)}"
                        f"  train_loss: {loss.item():.4f}"
                        f"  step_time: {time.time() - step_start:.4f}s"
                    )

            lr_scheduler.step()
            global_loss_sum = reduce_sum(epoch_loss_sum, context)
            global_step_count = reduce_sum(step_count, context)
            epoch_loss = global_loss_sum / max(global_step_count, 1.0)
            current_lr = lr_scheduler.get_last_lr()[0]
            log_main(context, f"  avg loss: {epoch_loss:.4f}  lr: {current_lr:.2e}")

            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": epoch_loss,
                        "train/lr": current_lr,
                        "epoch": epoch + 1,
                    }
                )

            if (epoch + 1) % args.val_interval == 0:
                barrier(context)
                if context.is_main_process:
                    model_for_inference.eval()
                    with torch.no_grad():
                        for val_data in val_loader:
                            val_inputs = val_data["image"].to(context.device)
                            val_labels = val_data["label"].to(context.device)
                            val_outputs = sliding_window_inference(
                                val_inputs,
                                roi_size=roi_size,
                                sw_batch_size=4,
                                predictor=model_for_inference,
                                overlap=0.5,
                            )
                            val_outputs = [
                                post_trans(i) for i in decollate_batch(val_outputs)
                            ]
                            dice_metric(y_pred=val_outputs, y=val_labels)
                            dice_metric_batch(y_pred=val_outputs, y=val_labels)

                        metric = dice_metric.aggregate().item()
                        metric_batch = dice_metric_batch.aggregate()
                        metric_tc = metric_batch[0].item()
                        metric_wt = metric_batch[1].item()
                        metric_et = metric_batch[2].item()
                        metric_values_tc.append(metric_tc)
                        metric_values_wt.append(metric_wt)
                        metric_values_et.append(metric_et)
                        dice_metric.reset()
                        dice_metric_batch.reset()

                        if metric > best_metric:
                            best_metric = metric
                            best_metric_epoch = epoch + 1
                            ckpt_path = os.path.join(save_dir, "best_metric_model.pth")
                            torch.save(model_for_inference.state_dict(), ckpt_path)
                            print(f"  -> saved new best model to {ckpt_path}")

                        if wandb_run is not None:
                            wandb.log(
                                {
                                    "val/dice_mean": metric,
                                    "val/dice_tc": metric_tc,
                                    "val/dice_wt": metric_wt,
                                    "val/dice_et": metric_et,
                                    "val/best_dice": best_metric,
                                    "epoch": epoch + 1,
                                }
                            )

                        print(
                            f"  val dice: {metric:.4f}"
                            f"  (TC={metric_tc:.4f}  WT={metric_wt:.4f}  ET={metric_et:.4f})"
                            f"\n  best dice: {best_metric:.4f} @ epoch {best_metric_epoch}"
                        )
                barrier(context)

            log_main(context, f"  epoch time: {time.time() - epoch_start:.1f}s")

        total_time = time.time() - total_start

        if context.is_main_process:
            print("=" * 60)
            print("Training complete")
            print(f"  Best mean Dice : {best_metric:.4f} @ epoch {best_metric_epoch}")
            if metric_values_tc:
                print(f"  Best TC Dice   : {max(metric_values_tc):.4f}")
                print(f"  Best WT Dice   : {max(metric_values_wt):.4f}")
                print(f"  Best ET Dice   : {max(metric_values_et):.4f}")
            print(f"  Total time     : {total_time:.1f}s ({total_time / 3600:.2f}h)")
            print(f"  Checkpoint     : {os.path.join(save_dir, 'best_metric_model.pth')}")
            print("=" * 60)
    finally:
        if wandb_run is not None:
            wandb.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
