#!/usr/bin/env python
"""BraTS2020 DynUNet trainer with configurable validation metrics."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from monai.config import print_config
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)
from monai.utils import set_determinism

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import ConvertToMultiChannelBasedOnBratsClassesd, EnsureFloatLabeld  # noqa: E402

load_dotenv()

DEFAULT_DATA_DIR = REPO_ROOT / "res" / "dataset" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
DEFAULT_SAVE_DIR = REPO_ROOT / "res" / "models"
WANDB_PROJECT = "cancervision"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "cancervision")
DEFAULT_SW_BATCH_SIZE = 4
DEFAULT_SPLIT_MANIFEST_NAME = "split_manifest.json"


@dataclass
class RuntimeContext:
    device: torch.device
    device_index: int | None
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


@dataclass(frozen=True)
class DistributedLaunchConfig:
    world_size: int
    rank: int
    local_rank: int
    distributed: bool
    allocated_gpu_count: int | None
    visible_gpu_count: int
    device_index: int | None


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
        default=None,
        help="DataLoader workers per process; defaults to a SLURM-aware safe value",
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
        "--label-schema",
        choices=("brats3", "identity"),
        default="brats3",
        help="Label interpretation: BraTS multi-label channels or use labels as provided",
    )
    parser.add_argument(
        "--compute-hd95",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute HD95 alongside Dice during validation",
    )
    parser.add_argument(
        "--hd95-space",
        choices=("metadata", "voxel"),
        default="metadata",
        help="Use physical spacing from metadata when available, otherwise fall back to voxel distances",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Optional JSON file describing the train/validation split to reuse or create",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=None,
        help="Optional voxel spacing to resample image and label volumes to before training",
    )
    parser.add_argument(
        "--orientation-axcodes",
        type=str,
        default=None,
        help="Optional orientation code (for example RAS) applied before training and validation",
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


def build_label_transform(label_schema: str):
    if label_schema == "brats3":
        return ConvertToMultiChannelBasedOnBratsClassesd(keys="label")
    if label_schema == "identity":
        return EnsureFloatLabeld(keys="label")
    raise ValueError(f"Unsupported label schema: {label_schema}")


def build_common_transforms(
    label_schema: str,
    *,
    target_spacing: Sequence[float] | None,
    orientation_axcodes: str | None,
) -> list[Any]:
    transforms: list[Any] = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    if orientation_axcodes:
        transforms.append(Orientationd(keys=["image", "label"], axcodes=orientation_axcodes))
    if target_spacing is not None:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(target_spacing),
                mode=("bilinear", "nearest"),
            )
        )
    transforms.extend(
        [
            build_label_transform(label_schema),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    return transforms


def get_train_transforms(
    roi_size: Sequence[int],
    num_samples: int,
    *,
    label_schema: str,
    target_spacing: Sequence[float] | None,
    orientation_axcodes: str | None,
) -> Compose:
    return Compose(
        [
            *build_common_transforms(
                label_schema,
                target_spacing=target_spacing,
                orientation_axcodes=orientation_axcodes,
            ),
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


def get_val_transforms(
    *,
    label_schema: str,
    target_spacing: Sequence[float] | None,
    orientation_axcodes: str | None,
) -> Compose:
    return Compose(
        build_common_transforms(
            label_schema,
            target_spacing=target_spacing,
            orientation_axcodes=orientation_axcodes,
        )
    )


def find_nifti(directory: str, pattern: str) -> str:
    for ext in (".nii", ".nii.gz"):
        candidate = os.path.join(directory, pattern + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find NIfTI file for pattern '{pattern}' in {directory}"
    )


def build_data_dicts(data_dir: str) -> list[dict[str, Any]]:
    data_dicts: list[dict[str, Any]] = []
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

        data_dicts.append(
            {
                "patient_id": patient_name,
                "image": [flair, t1, t1ce, t2],
                "label": seg,
            }
        )

    if not data_dicts:
        raise FileNotFoundError(f"No valid BraTS patient folders found in {data_dir}")

    return data_dicts


def build_model(out_channels: int = 3) -> DynUNet:
    return DynUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=out_channels,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320],
        dropout=0.2,
        res_block=True,
        deep_supervision=False,
    )


def infer_output_channels(
    sample: Mapping[str, Any],
    *,
    label_schema: str,
    target_spacing: Sequence[float] | None,
    orientation_axcodes: str | None,
) -> int:
    preview = Compose(
        build_common_transforms(
            label_schema,
            target_spacing=target_spacing,
            orientation_axcodes=orientation_axcodes,
        )
    )(dict(sample))
    label = preview["label"]
    if not isinstance(label, torch.Tensor):
        raise TypeError("Expected transformed label to be a tensor")
    if label.ndim < 2:
        raise ValueError(f"Expected channel-first label tensor, got shape {tuple(label.shape)}")
    return int(label.shape[0])


def class_names_for_schema(label_schema: str, output_channels: int) -> list[str]:
    if label_schema == "brats3":
        return ["tc", "wt", "et"]
    if output_channels == 1:
        return ["fg"]
    return [f"class_{index}" for index in range(output_channels)]


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return f"dynunet-brats-{job_id}"
    return f"dynunet-brats-{time.strftime('%Y%m%d-%H%M%S')}"


def resolve_split_manifest_path(split_manifest: str | None, run_dir: Path) -> Path:
    if split_manifest:
        return Path(split_manifest).expanduser().resolve()
    return run_dir / DEFAULT_SPLIT_MANIFEST_NAME


def save_split_manifest(
    path: Path,
    train_dicts: Sequence[Mapping[str, Any]],
    val_dicts: Sequence[Mapping[str, Any]],
    *,
    data_dir: str,
    seed: int,
    test_size: float,
    label_schema: str,
) -> None:
    payload = {
        "version": 1,
        "data_dir": data_dir,
        "seed": seed,
        "test_size": test_size,
        "label_schema": label_schema,
        "train_patient_ids": [str(item["patient_id"]) for item in train_dicts],
        "val_patient_ids": [str(item["patient_id"]) for item in val_dicts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(
    path: Path,
    data_dicts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data_by_patient_id: dict[str, dict[str, Any]] = {}
    for item in data_dicts:
        patient_id = str(item["patient_id"])
        if patient_id in data_by_patient_id:
            raise ValueError(f"Duplicate patient id in dataset discovery: {patient_id}")
        data_by_patient_id[patient_id] = dict(item)

    train_ids = payload.get("train_patient_ids")
    val_ids = payload.get("val_patient_ids")
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError(f"Invalid split manifest format: {path}")

    missing_ids = [
        patient_id
        for patient_id in [*train_ids, *val_ids]
        if patient_id not in data_by_patient_id
    ]
    if missing_ids:
        raise ValueError(
            f"Split manifest {path} references patients not present in the dataset: {missing_ids}"
        )

    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(f"Split manifest {path} contains train/val overlap: {sorted(overlap)}")

    train_dicts = [data_by_patient_id[patient_id] for patient_id in train_ids]
    val_dicts = [data_by_patient_id[patient_id] for patient_id in val_ids]
    return train_dicts, val_dicts


def is_main_process(context: RuntimeContext) -> bool:
    return context.rank == 0


def resolve_dataset_split(
    data_dicts: Sequence[Mapping[str, Any]],
    *,
    data_dir: str,
    run_dir: Path,
    args: argparse.Namespace,
    context: RuntimeContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path, bool]:
    split_manifest_path = resolve_split_manifest_path(args.split_manifest, run_dir)
    if split_manifest_path.is_file():
        train_dicts, val_dicts = load_split_manifest(split_manifest_path, data_dicts)
        return train_dicts, val_dicts, split_manifest_path, True

    train_raw, val_raw = train_test_split(
        list(data_dicts), test_size=args.test_size, random_state=args.seed
    )
    train_dicts = [dict(item) for item in train_raw]
    val_dicts = [dict(item) for item in val_raw]
    if is_main_process(context):
        save_split_manifest(
            split_manifest_path,
            train_dicts,
            val_dicts,
            data_dir=data_dir,
            seed=args.seed,
            test_size=args.test_size,
            label_schema=args.label_schema,
        )
    return train_dicts, val_dicts, split_manifest_path, False


def should_use_amp(args: argparse.Namespace, context: RuntimeContext) -> bool:
    return bool(args.amp and context.device.type == "cuda")


def _parse_env_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value.isdigit():
        return None
    return int(value)


def _count_csv_entries(value: str | None) -> int | None:
    if value is None:
        return None
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    return len(entries) if entries else None


def detect_allocated_gpu_count(env: Mapping[str, str] | None = None) -> int | None:
    env = os.environ if env is None else env
    slurm_gpus_on_node = _parse_env_int(env.get("SLURM_GPUS_ON_NODE"))
    if slurm_gpus_on_node is not None:
        return slurm_gpus_on_node

    slurm_step_gpus = _count_csv_entries(env.get("SLURM_STEP_GPUS"))
    if slurm_step_gpus is not None:
        return slurm_step_gpus

    cuda_visible_devices = _count_csv_entries(env.get("CUDA_VISIBLE_DEVICES"))
    if cuda_visible_devices is not None:
        return cuda_visible_devices

    return None


def resolve_num_workers(
    requested_num_workers: int | None,
    *,
    env: Mapping[str, str] | None = None,
    cpu_count: int | None = None,
) -> tuple[int, str | None]:
    env = os.environ if env is None else env
    host_cpu_count = cpu_count or os.cpu_count() or 1
    slurm_cpu_limit = _parse_env_int(env.get("SLURM_CPUS_PER_TASK"))

    available_cpu_budget = host_cpu_count
    if slurm_cpu_limit is not None and slurm_cpu_limit > 0:
        available_cpu_budget = min(available_cpu_budget, slurm_cpu_limit)

    if requested_num_workers is None:
        resolved_num_workers = min(4, available_cpu_budget)
        reason = (
            f"derived from available CPU budget={available_cpu_budget}"
            if available_cpu_budget
            else None
        )
        return max(resolved_num_workers, 0), reason

    resolved_num_workers = max(requested_num_workers, 0)
    if resolved_num_workers <= available_cpu_budget:
        return resolved_num_workers, None

    return (
        available_cpu_budget,
        (
            f"capped from requested {requested_num_workers} to {available_cpu_budget} "
            "to respect the available CPU budget"
        ),
    )


def _format_launch_env(env: Mapping[str, str]) -> str:
    keys = (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_GPUS_ON_NODE",
        "SLURM_STEP_GPUS",
        "CUDA_VISIBLE_DEVICES",
    )
    return ", ".join(f"{key}={env.get(key, '<unset>')}" for key in keys)


def resolve_distributed_launch_config(
    env: Mapping[str, str] | None = None,
    *,
    cuda_available: bool,
    visible_gpu_count: int,
) -> DistributedLaunchConfig:
    env = os.environ if env is None else env

    world_size = int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", "1")))
    rank = int(env.get("RANK", env.get("SLURM_PROCID", "0")))
    local_rank = int(env.get("LOCAL_RANK", env.get("SLURM_LOCALID", "0")))
    distributed = world_size > 1
    allocated_gpu_count = detect_allocated_gpu_count(env)

    if not distributed:
        return DistributedLaunchConfig(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed=False,
            allocated_gpu_count=allocated_gpu_count,
            visible_gpu_count=visible_gpu_count,
            device_index=None,
        )

    if not cuda_available:
        raise RuntimeError("Distributed training requires CUDA for this trainer.")

    if allocated_gpu_count is not None and allocated_gpu_count < world_size:
        raise RuntimeError(
            "Distributed launch misconfigured: allocated GPU count is smaller than "
            f"world size (world_size={world_size}, allocated_gpu_count={allocated_gpu_count}, "
            f"rank={rank}, local_rank={local_rank}). Relevant env: {_format_launch_env(env)}"
        )

    if visible_gpu_count < 1:
        raise RuntimeError(
            "Distributed training requires at least one CUDA-visible GPU per process "
            f"(visible_gpu_count={visible_gpu_count}, rank={rank}, local_rank={local_rank}, "
            f"allocated_gpu_count={allocated_gpu_count}). Relevant env: {_format_launch_env(env)}"
        )

    if visible_gpu_count == 1 and (
        allocated_gpu_count is None or allocated_gpu_count >= world_size
    ):
        device_index = 0
    elif local_rank < visible_gpu_count:
        device_index = local_rank
    else:
        raise RuntimeError(
            "Unable to map local rank to a CUDA device "
            f"(rank={rank}, local_rank={local_rank}, world_size={world_size}, "
            f"visible_gpu_count={visible_gpu_count}, allocated_gpu_count={allocated_gpu_count}). "
            f"Relevant env: {_format_launch_env(env)}"
        )

    return DistributedLaunchConfig(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed=True,
        allocated_gpu_count=allocated_gpu_count,
        visible_gpu_count=visible_gpu_count,
        device_index=device_index,
    )


def setup_device_and_distributed() -> RuntimeContext:
    cuda_available = torch.cuda.is_available()
    visible_gpu_count = torch.cuda.device_count() if cuda_available else 0
    launch_config = resolve_distributed_launch_config(
        cuda_available=cuda_available,
        visible_gpu_count=visible_gpu_count,
    )

    if launch_config.distributed:
        os.environ.setdefault("WORLD_SIZE", str(launch_config.world_size))
        os.environ.setdefault("RANK", str(launch_config.rank))
        os.environ.setdefault("LOCAL_RANK", str(launch_config.local_rank))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        assert launch_config.device_index is not None
        torch.cuda.set_device(launch_config.device_index)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=launch_config.rank,
            world_size=launch_config.world_size,
        )
        device = torch.device("cuda", launch_config.device_index)
        device_index = launch_config.device_index
    else:
        if cuda_available:
            device = torch.device("cuda:0")
            device_index = 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_index = None
        else:
            device = torch.device("cpu")
            device_index = None

    return RuntimeContext(
        device=device,
        device_index=device_index,
        distributed=launch_config.distributed,
        rank=launch_config.rank,
        local_rank=launch_config.local_rank,
        world_size=launch_config.world_size,
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


def get_autocast_context(
    context: RuntimeContext, enabled: bool
) -> contextlib.AbstractContextManager:
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
    scaler: torch.amp.GradScaler | None,
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
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    best_metric_epoch: int,
) -> None:
    model_state = (
        model.module.state_dict()
        if isinstance(model, DistributedDataParallel)
        else model.state_dict()
    )
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


def _coerce_sequence(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        try:
            converted = value.tolist()
        except TypeError:
            converted = None
        if converted is not None:
            return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def normalize_spacing_value(value: Any, spatial_dims: int) -> list[float] | None:
    sequence = _coerce_sequence(value)
    if sequence is None:
        return None
    if sequence and isinstance(sequence[0], (list, tuple, torch.Tensor)):
        nested = _coerce_sequence(sequence[0])
        sequence = nested if nested is not None else sequence
    if len(sequence) == 1:
        return [float(sequence[0])] * spatial_dims
    if len(sequence) >= spatial_dims + 1:
        sequence = sequence[1 : spatial_dims + 1]
    elif len(sequence) >= spatial_dims:
        sequence = sequence[:spatial_dims]
    else:
        return None
    return [float(item) for item in sequence]


def spacing_from_affine(affine: Any, spatial_dims: int) -> list[float] | None:
    if affine is None:
        return None
    affine_tensor = torch.as_tensor(affine, dtype=torch.float32)
    if affine_tensor.ndim == 3:
        affine_tensor = affine_tensor[0]
    if affine_tensor.shape[0] < spatial_dims or affine_tensor.shape[1] < spatial_dims:
        return None
    spatial_affine = affine_tensor[:spatial_dims, :spatial_dims]
    return [
        float(torch.linalg.vector_norm(spatial_affine[:, axis]).item())
        for axis in range(spatial_dims)
    ]


def extract_image_spacing(image: Any, spatial_dims: int) -> list[float] | None:
    metadata_candidates: list[Mapping[str, Any]] = []
    meta = getattr(image, "meta", None)
    if isinstance(meta, Mapping):
        metadata_candidates.append(meta)
    image_meta_dict = getattr(image, "meta_dict", None)
    if isinstance(image_meta_dict, Mapping):
        metadata_candidates.append(image_meta_dict)

    for metadata in metadata_candidates:
        for key in ("spacing", "pixdim"):
            spacing = normalize_spacing_value(metadata.get(key), spatial_dims)
            if spacing is not None:
                return spacing
        for key in ("affine", "original_affine"):
            spacing = spacing_from_affine(metadata.get(key), spatial_dims)
            if spacing is not None:
                return spacing
    return None


def summarize_segmentation_metrics(
    predictions: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    class_names: Sequence[str],
    *,
    compute_hd95: bool,
    spacings: Sequence[Sequence[float] | None] | None = None,
) -> dict[str, float]:
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same number of samples")
    if not predictions:
        raise ValueError("Validation requires at least one prediction")
    if spacings is not None and len(spacings) != len(predictions):
        raise ValueError("Spacing metadata must align with the number of predictions")

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    dice_metric(y_pred=list(predictions), y=list(labels))
    dice_metric_batch(y_pred=list(predictions), y=list(labels))

    dice_mean = float(dice_metric.aggregate().item())
    dice_batch = dice_metric_batch.aggregate().detach().cpu().flatten().tolist()
    dice_metric.reset()
    dice_metric_batch.reset()

    case_count = len(predictions)
    gt_positive_case_count = [0] * len(class_names)
    pred_positive_case_count = [0] * len(class_names)
    empty_both_case_count = [0] * len(class_names)
    missed_case_count = [0] * len(class_names)
    pred_only_case_count = [0] * len(class_names)
    hd95_valid_case_count = [0] * len(class_names)
    gt_voxel_count = [0.0] * len(class_names)
    total_voxel_count = [0] * len(class_names)
    hd95_values: list[list[float]] = [[] for _ in class_names]

    for case_index, (prediction, label) in enumerate(zip(predictions, labels)):
        spacing = spacings[case_index] if spacings is not None else None
        for class_index, _class_name in enumerate(class_names):
            pred_channel = prediction[class_index] > 0
            label_channel = label[class_index] > 0
            pred_has_foreground = bool(pred_channel.any().item())
            label_has_foreground = bool(label_channel.any().item())

            pred_positive_case_count[class_index] += int(pred_has_foreground)
            gt_positive_case_count[class_index] += int(label_has_foreground)
            gt_voxel_count[class_index] += float(label_channel.sum().item())
            total_voxel_count[class_index] += int(label_channel.numel())

            if not pred_has_foreground and not label_has_foreground:
                empty_both_case_count[class_index] += 1
                continue
            if label_has_foreground and not pred_has_foreground:
                missed_case_count[class_index] += 1
                continue
            if pred_has_foreground and not label_has_foreground:
                pred_only_case_count[class_index] += 1
                continue

            if not compute_hd95:
                continue

            hd95_value = float(
                compute_hausdorff_distance(
                    pred_channel.unsqueeze(0).unsqueeze(0).float(),
                    label_channel.unsqueeze(0).unsqueeze(0).float(),
                    include_background=True,
                    percentile=95,
                    spacing=spacing,
                )[0, 0].item()
            )
            if math.isfinite(hd95_value):
                hd95_values[class_index].append(hd95_value)
                hd95_valid_case_count[class_index] += 1

    metrics: dict[str, float] = {
        "dice_mean": dice_mean,
        "val_case_count": float(case_count),
    }

    for class_index, class_name in enumerate(class_names):
        metrics[f"dice_{class_name}"] = float(dice_batch[class_index])
        metrics[f"gt_case_prevalence_{class_name}"] = (
            gt_positive_case_count[class_index] / case_count
        )
        metrics[f"pred_case_prevalence_{class_name}"] = (
            pred_positive_case_count[class_index] / case_count
        )
        metrics[f"gt_voxel_prevalence_{class_name}"] = (
            gt_voxel_count[class_index] / max(total_voxel_count[class_index], 1)
        )
        metrics[f"empty_both_case_count_{class_name}"] = float(
            empty_both_case_count[class_index]
        )
        metrics[f"missed_case_count_{class_name}"] = float(
            missed_case_count[class_index]
        )
        metrics[f"pred_only_case_count_{class_name}"] = float(
            pred_only_case_count[class_index]
        )
        metrics[f"hd95_valid_case_count_{class_name}"] = float(
            hd95_valid_case_count[class_index]
        )

    if compute_hd95:
        valid_hd95 = [value for per_class in hd95_values for value in per_class]
        metrics["hd95_mean"] = (
            float(sum(valid_hd95) / len(valid_hd95)) if valid_hd95 else float("nan")
        )
        for class_index, class_name in enumerate(class_names):
            per_class_values = hd95_values[class_index]
            metrics[f"hd95_{class_name}"] = (
                float(sum(per_class_values) / len(per_class_values))
                if per_class_values
                else float("nan")
            )

    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_function: DiceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
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
    class_names: Sequence[str],
) -> dict[str, float] | None:
    if not is_main_process(context) or val_loader is None:
        return None

    predictor = model.module if isinstance(model, DistributedDataParallel) else model
    predictor.eval()

    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    spacings: list[list[float] | None] = []
    hd95_voxel_fallback_cases = 0

    with torch.no_grad():
        for val_data in val_loader:
            image_items = decollate_batch(val_data["image"])
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
            predictions.extend(val_outputs_list)
            labels.extend(val_labels_list)

            if args.hd95_space == "metadata":
                for image_item, label_item in zip(image_items, val_labels_list):
                    spacing = extract_image_spacing(image_item, spatial_dims=label_item.ndim - 1)
                    if spacing is None:
                        hd95_voxel_fallback_cases += 1
                    spacings.append(spacing)

    metrics = summarize_segmentation_metrics(
        predictions,
        labels,
        class_names,
        compute_hd95=args.compute_hd95,
        spacings=spacings if args.hd95_space == "metadata" else None,
    )
    if args.compute_hd95:
        metrics["hd95_voxel_fallback_cases"] = float(hd95_voxel_fallback_cases)
    return metrics


def format_metric_summary(
    metrics: Mapping[str, float],
    class_names: Sequence[str],
    *,
    metric_prefix: str,
) -> str:
    parts = []
    for class_name in class_names:
        key = f"{metric_prefix}_{class_name}"
        value = metrics.get(key, float("nan"))
        rendered = "nan" if math.isnan(value) else f"{value:.4f}"
        parts.append(f"{class_name.upper()}={rendered}")
    return " ".join(parts)


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
        rank0_print(
            context,
            f"Distributed: {context.distributed} (world_size={context.world_size})",
        )
        rank0_print(context, f"Run directory: {run_dir}")

        data_dir = os.path.normpath(args.data_dir)
        rank0_print(context, f"Data directory : {data_dir}")
        rank0_print(context, f"Exists         : {os.path.isdir(data_dir)}")

        data_dicts = build_data_dicts(data_dir)
        rank0_print(context, f"Total patients : {len(data_dicts)}")

        train_dicts, val_dicts, split_manifest_path, reused_split = resolve_dataset_split(
            data_dicts,
            data_dir=data_dir,
            run_dir=run_dir,
            args=args,
            context=context,
        )
        synchronize(context)
        rank0_print(context, f"Split manifest : {split_manifest_path}")
        rank0_print(context, f"Split reused   : {reused_split}")
        rank0_print(context, f"Train patients : {len(train_dicts)}")
        rank0_print(context, f"Val patients   : {len(val_dicts)}")

        if not train_dicts and not val_dicts:
            raise RuntimeError("No training or validation samples were discovered")

        preview_sample = train_dicts[0] if train_dicts else val_dicts[0]
        output_channels = infer_output_channels(
            preview_sample,
            label_schema=args.label_schema,
            target_spacing=args.target_spacing,
            orientation_axcodes=args.orientation_axcodes,
        )
        class_names = class_names_for_schema(args.label_schema, output_channels)
        rank0_print(context, f"Label schema   : {args.label_schema}")
        rank0_print(context, f"Output channels: {output_channels} ({', '.join(class_names)})")

        roi_size = tuple(args.roi_size)
        train_ds = Dataset(
            data=train_dicts,
            transform=get_train_transforms(
                roi_size,
                args.num_samples,
                label_schema=args.label_schema,
                target_spacing=args.target_spacing,
                orientation_axcodes=args.orientation_axcodes,
            ),
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

        num_workers, worker_reason = resolve_num_workers(args.num_workers)
        if worker_reason:
            rank0_print(context, f"DataLoader workers: {num_workers} ({worker_reason})")
        else:
            rank0_print(context, f"DataLoader workers: {num_workers}")
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
            val_ds = Dataset(
                data=val_dicts,
                transform=get_val_transforms(
                    label_schema=args.label_schema,
                    target_spacing=args.target_spacing,
                    orientation_axcodes=args.orientation_axcodes,
                ),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=pin_memory,
            )

        model = build_model(out_channels=output_channels).to(context.device)
        if context.distributed:
            assert context.device_index is not None
            model = DistributedDataParallel(
                model,
                device_ids=[context.device_index],
                output_device=context.device_index,
            )

        loss_function = DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )
        scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
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
            "label_schema": args.label_schema,
            "compute_hd95": args.compute_hd95,
            "hd95_space": args.hd95_space,
            "split_manifest": str(split_manifest_path),
            "target_spacing": list(args.target_spacing) if args.target_spacing else None,
            "orientation_axcodes": args.orientation_axcodes,
            "output_channels": output_channels,
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
                    class_names=class_names,
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

                    wandb_payload = {
                        f"val/{key}": value for key, value in metrics.items()
                    }
                    wandb_payload["val/best_dice"] = best_metric
                    wandb_payload["epoch"] = epoch + 1
                    wandb_run.log(wandb_payload)

                    print(
                        f"  val dice: {metrics['dice_mean']:.4f}"
                        f"  ({format_metric_summary(metrics, class_names, metric_prefix='dice')})",
                        flush=True,
                    )
                    if args.compute_hd95:
                        hd95_mean = metrics.get("hd95_mean", float("nan"))
                        hd95_rendered = "nan" if math.isnan(hd95_mean) else f"{hd95_mean:.4f}"
                        print(
                            f"  val hd95: {hd95_rendered}"
                            f"  ({format_metric_summary(metrics, class_names, metric_prefix='hd95')})",
                            flush=True,
                        )
                    print(
                        "  gt prevalence: "
                        + " ".join(
                            f"{class_name.upper()}={metrics[f'gt_case_prevalence_{class_name}']:.2%}"
                            for class_name in class_names
                        ),
                        flush=True,
                    )
                    print(
                        "  missed cases: "
                        + " ".join(
                            f"{class_name.upper()}={int(metrics[f'missed_case_count_{class_name}'])}"
                            for class_name in class_names
                        ),
                        flush=True,
                    )
                    print(
                        f"  best dice: {best_metric:.4f} @ epoch {best_metric_epoch}",
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
            print(
                f"  Total time     : {total_time:.1f}s ({total_time / 3600:.2f}h)",
                flush=True,
            )
            print(f"  Split manifest : {split_manifest_path}", flush=True)
            print(f"  Checkpoint     : {run_dir / 'best_metric_model.pth'}", flush=True)
            print("=" * 60, flush=True)
    finally:
        wandb_run.finish()
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
