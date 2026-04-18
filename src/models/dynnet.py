#!/usr/bin/env python
"""BraTS DynUNet trainer for 2020/2023/2024 NIfTI layouts."""

from __future__ import annotations

import argparse
import csv
import contextlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from sklearn.model_selection import train_test_split

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

from src.datasets import (  # noqa: E402
    BinarizeLabeld,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureFloatLabeld,
    build_brats_data_dicts,
    default_brats_data_dir,
)
from src.datasets.standardize.pathing import resolve_existing_path  # noqa: E402

load_dotenv()

DEFAULT_DATA_DIR = default_brats_data_dir(REPO_ROOT)
DEFAULT_CANCERVISION_DATASET_ROOT = (
    REPO_ROOT / "res" / "dataset" / "cancervision-standardization"
)
DEFAULT_CANCERVISION_TASK_MANIFEST = (
    DEFAULT_CANCERVISION_DATASET_ROOT
    / "task_manifests"
    / "segmentation_binary_curated.csv"
)
DEFAULT_SAVE_DIR = REPO_ROOT / "res" / "models"
WANDB_PROJECT = "cancervision"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "cancervision")
DEFAULT_ROI_SIZE = (96, 96, 96)
DEFAULT_NUM_SAMPLES = 1
DEFAULT_VAL_SW_BATCH_SIZE = 1
WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")


@dataclass
class RuntimeContext:
    device: torch.device
    device_index: int | None
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


class NoOpWandbRun:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    in_channels: int
    out_channels: int
    metric_names: tuple[str, ...]
    train_transform_builder: Callable[[Sequence[int], int], Compose]
    val_transform_builder: Callable[[], Compose]


@dataclass(frozen=True)
class GpuProfileConfig:
    name: str
    roi_size: tuple[int, int, int]
    num_samples: int
    model_filters: tuple[int, int, int, int, int]
    val_sw_batch_size: int = DEFAULT_VAL_SW_BATCH_SIZE


GPU_PROFILE_CONFIGS: dict[str, GpuProfileConfig] = {
    "gpu16g": GpuProfileConfig(
        name="gpu16g",
        roi_size=(64, 64, 64),
        num_samples=1,
        model_filters=(16, 32, 64, 128, 192),
    ),
    "gpu32g": GpuProfileConfig(
        name="gpu32g",
        roi_size=(80, 80, 80),
        num_samples=1,
        model_filters=(24, 48, 96, 192, 256),
    ),
    "gpu40g": GpuProfileConfig(
        name="gpu40g",
        roi_size=DEFAULT_ROI_SIZE,
        num_samples=DEFAULT_NUM_SAMPLES,
        model_filters=(32, 64, 128, 256, 320),
    ),
    "gpu80g": GpuProfileConfig(
        name="gpu80g",
        roi_size=(128, 128, 128),
        num_samples=1,
        model_filters=(32, 64, 128, 256, 320),
    ),
}
DEFAULT_GPU_PROFILE_NAME = "gpu40g"
GPU_PROFILE_ALIASES = {
    "p100": "gpu16g",
    "v100": "gpu32g",
    "a100": "gpu40g",
    "gpu16g": "gpu16g",
    "gpu32g": "gpu32g",
    "gpu40g": "gpu40g",
    "gpu80g": "gpu80g",
    "sxm4": "gpu80g",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a MONAI DynUNet on BraTS 2020/2023/2024 volumes or "
            "CancerVision standardized segmentation task manifests"
        )
    )
    parser.add_argument(
        "--gpu-profile",
        choices=("auto", *GPU_PROFILE_CONFIGS.keys(), "p100", "v100", "a100", "sxm4"),
        default="auto",
        help=(
            "Memory preset for DynUNet training. `auto` uses Slurm constraints or "
            "detected GPU memory."
        ),
    )
    parser.add_argument(
        "--dataset-source",
        choices=("brats", "cancervision_binary_seg"),
        default="brats",
        help="Input pipeline to use: raw BraTS folders or CancerVision segmentation task manifest.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.normpath(str(DEFAULT_DATA_DIR)),
        help="Path to BraTS 2020/2023/2024 root containing patient dirs.",
    )
    parser.add_argument(
        "--task-manifest",
        type=str,
        default=os.path.normpath(str(DEFAULT_CANCERVISION_TASK_MANIFEST)),
        help=(
            "Path to CancerVision segmentation task manifest CSV. Used when "
            "--dataset-source=cancervision_binary_seg."
        ),
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
        default=None,
        help="Random crop ROI size as three integers",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of random crops per volume per iteration",
    )
    parser.add_argument(
        "--model-filters",
        type=int,
        nargs=5,
        default=None,
        help="DynUNet filter widths for downsample levels.",
    )
    parser.add_argument(
        "--train-micro-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum samples/crops per optimizer micro-step after MONAI collation. "
            "Use 1 to reduce CUDA memory pressure for 3D volumes."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA AMP during training and validation",
    )
    parser.add_argument(
        "--val-sw-batch-size",
        type=int,
        default=None,
        help="Sliding-window batch size used during validation inference",
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


def build_data_dicts(data_dir: str) -> list[dict[str, list[str] | str]]:
    return build_brats_data_dicts(data_dir)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _looks_like_windows_drive_path(path_text: str) -> bool:
    return bool(WINDOWS_DRIVE_PATTERN.match(path_text))


def _resolve_manifest_data_path(
    raw_path: str,
    *,
    manifest_dir: Path,
    field_name: str,
    case_id: str,
) -> str:
    candidates: list[str | Path] = []
    if raw_path:
        if Path(raw_path).is_absolute() or _looks_like_windows_drive_path(raw_path):
            candidates.append(raw_path)
        else:
            candidates.append(manifest_dir / raw_path)
            candidates.append(raw_path)

    for candidate in candidates:
        try:
            return os.path.normpath(str(resolve_existing_path(candidate)))
        except FileNotFoundError:
            continue

    joined_candidates = ", ".join(str(candidate) for candidate in candidates) or "<empty>"
    raise FileNotFoundError(
        f"Missing {field_name} for manifest case '{case_id}'. Checked: {joined_candidates}"
    )


def build_cancervision_segmentation_splits(
    task_manifest_path: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    manifest_path = Path(task_manifest_path)
    rows = _read_csv_rows(manifest_path)
    manifest_dir = manifest_path.parent
    allowed_splits = {"train", "val", "test"}
    seen_case_ids: set[str] = set()
    split_rows = {split: [] for split in allowed_splits}

    for index, row in enumerate(rows, start=1):
        if row.get("exclude_reason"):
            continue
        if not row.get("image_path") or not row.get("mask_path"):
            continue

        split_name = (row.get("task_split") or "").strip().lower()
        case_id = (
            row.get("global_case_id")
            or row.get("subject_id")
            or row.get("image_path")
            or f"row_{index}"
        )
        if split_name not in allowed_splits:
            raise RuntimeError(
                f"Unsupported task_split '{row.get('task_split', '')}' for case '{case_id}' "
                f"in manifest {manifest_path}. Expected one of {sorted(allowed_splits)}."
            )
        if case_id in seen_case_ids:
            raise RuntimeError(
                f"Duplicate case '{case_id}' found in CancerVision task manifest: {manifest_path}"
            )
        seen_case_ids.add(case_id)

        split_rows[split_name].append(
            {
                "image": _resolve_manifest_data_path(
                    row["image_path"],
                    manifest_dir=manifest_dir,
                    field_name="image_path",
                    case_id=case_id,
                ),
                "label": _resolve_manifest_data_path(
                    row["mask_path"],
                    manifest_dir=manifest_dir,
                    field_name="mask_path",
                    case_id=case_id,
                ),
            }
        )

    train_rows = split_rows["train"]
    val_rows = split_rows["val"]
    test_rows = split_rows["test"]
    return train_rows, val_rows, test_rows


def get_brats_train_transforms(roi_size: Sequence[int], num_samples: int) -> Compose:
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


def get_brats_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="label"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def get_cancervision_binary_seg_train_transforms(
    roi_size: Sequence[int],
    num_samples: int,
) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            BinarizeLabeld(keys="label"),
            EnsureFloatLabeld(keys="label"),
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


def get_cancervision_binary_seg_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            BinarizeLabeld(keys="label"),
            EnsureFloatLabeld(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )


def get_dataset_config(dataset_source: str) -> DatasetConfig:
    if dataset_source == "brats":
        return DatasetConfig(
            name="brats",
            in_channels=4,
            out_channels=3,
            metric_names=("tc", "wt", "et"),
            train_transform_builder=get_brats_train_transforms,
            val_transform_builder=get_brats_val_transforms,
        )
    if dataset_source == "cancervision_binary_seg":
        return DatasetConfig(
            name="cancervision_binary_seg",
            in_channels=1,
            out_channels=1,
            metric_names=("lesion",),
            train_transform_builder=get_cancervision_binary_seg_train_transforms,
            val_transform_builder=get_cancervision_binary_seg_val_transforms,
        )
    raise ValueError(f"Unsupported dataset source: {dataset_source}")


def build_model(*, in_channels: int = 4, out_channels: int = 3) -> DynUNet:
    return build_model_with_filters(
        in_channels=in_channels,
        out_channels=out_channels,
        filters=GPU_PROFILE_CONFIGS[DEFAULT_GPU_PROFILE_NAME].model_filters,
    )


def build_model_with_filters(
    *,
    in_channels: int = 4,
    out_channels: int = 3,
    filters: Sequence[int],
) -> DynUNet:
    return DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        filters=list(filters),
        dropout=0.2,
        res_block=True,
        deep_supervision=False,
    )


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    job_id = os.environ.get("SLURM_JOB_ID")
    run_prefix = (
        "dynunet-cancervision"
        if args.dataset_source == "cancervision_binary_seg"
        else "dynunet-brats"
    )
    if job_id:
        return f"{run_prefix}-{job_id}"
    return f"{run_prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def is_main_process(context: RuntimeContext) -> bool:
    return context.rank == 0


def should_use_amp(args: argparse.Namespace, context: RuntimeContext) -> bool:
    return bool(args.amp and context.device.type == "cuda")


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


def normalize_gpu_profile_name(profile_name: str) -> str:
    normalized = profile_name.strip().lower()
    try:
        return GPU_PROFILE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported GPU profile '{profile_name}'") from exc


def detect_gpu_profile_from_constraints(
    env: Mapping[str, str] | None = None,
) -> str | None:
    env = os.environ if env is None else env
    raw_constraints = " ".join(
        value
        for value in (
            env.get("SLURM_JOB_CONSTRAINTS"),
            env.get("SLURM_CONSTRAINT"),
            env.get("SBATCH_CONSTRAINT"),
        )
        if value
    ).lower()
    if not raw_constraints:
        return None

    for profile_name in ("gpu80g", "gpu40g", "gpu32g", "gpu16g"):
        if profile_name in raw_constraints:
            return profile_name
    for alias in ("sxm4", "a100", "v100", "p100"):
        if alias in raw_constraints:
            return normalize_gpu_profile_name(alias)
    return None


def detect_gpu_profile_from_device(context: RuntimeContext) -> str | None:
    if context.device.type != "cuda":
        return None
    total_memory_gib = (
        torch.cuda.get_device_properties(context.device).total_memory / (1024**3)
    )
    if total_memory_gib <= 20:
        return "gpu16g"
    if total_memory_gib <= 36:
        return "gpu32g"
    if total_memory_gib <= 48:
        return "gpu40g"
    return "gpu80g"


def resolve_gpu_profile(
    requested_profile: str,
    context: RuntimeContext,
    env: Mapping[str, str] | None = None,
) -> GpuProfileConfig:
    if requested_profile != "auto":
        return GPU_PROFILE_CONFIGS[normalize_gpu_profile_name(requested_profile)]

    detected_from_constraints = detect_gpu_profile_from_constraints(env)
    if detected_from_constraints is not None:
        return GPU_PROFILE_CONFIGS[detected_from_constraints]

    detected_from_device = detect_gpu_profile_from_device(context)
    if detected_from_device is not None:
        return GPU_PROFILE_CONFIGS[detected_from_device]

    return GPU_PROFILE_CONFIGS[DEFAULT_GPU_PROFILE_NAME]


def apply_gpu_profile_defaults(
    args: argparse.Namespace,
    context: RuntimeContext,
    env: Mapping[str, str] | None = None,
) -> GpuProfileConfig:
    profile = resolve_gpu_profile(args.gpu_profile, context, env)
    if args.roi_size is None:
        args.roi_size = list(profile.roi_size)
    if args.num_samples is None:
        args.num_samples = profile.num_samples
    if args.model_filters is None:
        args.model_filters = list(profile.model_filters)
    if args.val_sw_batch_size is None:
        args.val_sw_batch_size = profile.val_sw_batch_size
    return profile


def detect_requested_world_size(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    return int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", "1")))


def setup_device_and_distributed() -> RuntimeContext:
    env = os.environ
    requested_world_size = detect_requested_world_size(env)
    rank = int(env.get("RANK", env.get("SLURM_PROCID", "0")))
    local_rank = int(env.get("LOCAL_RANK", env.get("SLURM_LOCALID", "0")))
    if requested_world_size > 1 or rank != 0 or local_rank != 0:
        raise RuntimeError(
            "This trainer supports single-process single-GPU runs only. "
            f"Launch with one task and one GPU. Relevant env: {_format_launch_env(env)}"
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
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
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
    )


def cleanup_distributed(context: RuntimeContext) -> None:
    return None


def synchronize(context: RuntimeContext) -> None:
    return None


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


def build_micro_batch_slices(total_batch_size: int, requested_micro_batch_size: int) -> list[slice]:
    if total_batch_size < 1:
        raise ValueError("total_batch_size must be at least 1")
    if requested_micro_batch_size < 1:
        raise ValueError("requested_micro_batch_size must be at least 1")

    micro_batch_size = min(total_batch_size, requested_micro_batch_size)
    return [
        slice(start, min(start + micro_batch_size, total_batch_size))
        for start in range(0, total_batch_size, micro_batch_size)
    ]


def reduce_mean(value: float, count: int, context: RuntimeContext) -> float:
    return value / max(count, 1)


def build_dataset_splits(
    args: argparse.Namespace,
) -> tuple[list[dict[str, list[str] | str]], list[dict[str, list[str] | str]], list[dict[str, list[str] | str]]]:
    if args.dataset_source == "brats":
        data_dir = os.path.normpath(args.data_dir)
        data_dicts = build_data_dicts(data_dir)
        train_dicts, val_dicts = train_test_split(
            data_dicts, test_size=args.test_size, random_state=args.seed
        )
        return train_dicts, val_dicts, []

    if args.dataset_source == "cancervision_binary_seg":
        task_manifest = os.path.normpath(args.task_manifest)
        train_dicts, val_dicts, test_dicts = build_cancervision_segmentation_splits(
            task_manifest
        )
        if not train_dicts:
            raise RuntimeError(
                f"No train rows found in CancerVision task manifest: {task_manifest}"
            )
        if not val_dicts:
            raise RuntimeError(
                f"No val rows found in CancerVision task manifest: {task_manifest}"
            )
        return train_dicts, val_dicts, test_dicts

    raise ValueError(f"Unsupported dataset source: {args.dataset_source}")


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
    model_to_load = model

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
    model_state = model.state_dict()
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
) -> tuple[float, float]:
    model.train()
    epoch_loss = 0.0
    step_count = 0

    for step, batch_data in enumerate(train_loader, start=1):
        step_start = time.time()
        step_count += 1
        inputs = batch_data["image"]
        labels = batch_data["label"]
        micro_batch_slices = build_micro_batch_slices(
            total_batch_size=inputs.shape[0],
            requested_micro_batch_size=args.train_micro_batch_size,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = None
        step_loss = 0.0
        # Keep full collated batch on CPU and move one micro-batch at a time to the GPU.
        # This lets us spend memory on larger 3D crops instead of staging every crop at once.
        for batch_slice in micro_batch_slices:
            micro_inputs = inputs[batch_slice].to(context.device, non_blocking=True)
            micro_labels = labels[batch_slice].to(context.device, non_blocking=True)
            micro_weight = micro_inputs.shape[0] / inputs.shape[0]

            with get_autocast_context(context, should_use_amp(args, context)):
                outputs = model(micro_inputs)
                micro_loss = loss_function(outputs, micro_labels)
                scaled_loss = micro_loss * micro_weight

            step_loss += micro_loss.item() * micro_weight
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            loss = micro_loss

        if loss is None:
            raise RuntimeError("Training batch produced no micro-batches.")

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        epoch_loss += step_loss
        if is_main_process(context) and step % 10 == 0:
            print(
                f"  step {step}/{len(train_loader)}"
                f"  train_loss: {step_loss:.4f}"
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
    dataset_config: DatasetConfig,
) -> dict[str, float] | None:
    if not is_main_process(context) or val_loader is None:
        return None

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    model.eval()

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(context.device, non_blocking=True)
            val_labels = val_data["label"].to(context.device, non_blocking=True)
            with get_autocast_context(context, should_use_amp(args, context)):
                val_outputs = sliding_window_inference(
                    val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=args.val_sw_batch_size,
                    predictor=model,
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

    metrics = {"dice_mean": metric}
    for index, metric_name in enumerate(dataset_config.metric_names):
        metrics[f"dice_{metric_name}"] = metric_batch[index].item()
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = setup_device_and_distributed()
    if context.device.type == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()
    gpu_profile = apply_gpu_profile_defaults(args, context)
    if args.train_micro_batch_size < 1:
        raise ValueError("--train-micro-batch-size must be at least 1")
    if args.val_sw_batch_size < 1:
        raise ValueError("--val-sw-batch-size must be at least 1")
    if len(args.roi_size) != 3 or any(size < 32 for size in args.roi_size):
        raise ValueError("--roi-size must contain three integers >= 32")
    if len(args.model_filters) != 5 or any(width < 8 for width in args.model_filters):
        raise ValueError("--model-filters must contain five integers >= 8")

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
        dataset_config = get_dataset_config(args.dataset_source)

        rank0_print(context, f"Using device: {context.device}")
        rank0_print(context, f"Distributed: {context.distributed} (world_size={context.world_size})")
        rank0_print(context, f"GPU profile    : {gpu_profile.name}")
        rank0_print(context, f"Run directory: {run_dir}")
        rank0_print(context, f"Dataset source : {args.dataset_source}")
        if args.dataset_source == "brats":
            data_dir = os.path.normpath(args.data_dir)
            rank0_print(context, f"Data directory : {data_dir}")
            rank0_print(context, f"Exists         : {os.path.isdir(data_dir)}")
        else:
            task_manifest = os.path.normpath(args.task_manifest)
            rank0_print(context, f"Task manifest  : {task_manifest}")
            rank0_print(context, f"Exists         : {os.path.isfile(task_manifest)}")
        rank0_print(context, f"Train micro-batch size: {args.train_micro_batch_size}")
        rank0_print(context, f"Validation sw_batch_size: {args.val_sw_batch_size}")
        rank0_print(context, f"ROI size       : {tuple(args.roi_size)}")
        rank0_print(context, f"Num samples    : {args.num_samples}")
        rank0_print(context, f"Model filters  : {tuple(args.model_filters)}")

        train_dicts, val_dicts, test_dicts = build_dataset_splits(args)
        rank0_print(
            context,
            f"Train cases    : {len(train_dicts)}",
        )
        rank0_print(context, f"Val cases      : {len(val_dicts)}")
        if args.dataset_source == "cancervision_binary_seg":
            rank0_print(context, f"Test cases     : {len(test_dicts)}")

        roi_size = tuple(args.roi_size)
        train_ds = Dataset(
            data=train_dicts,
            transform=dataset_config.train_transform_builder(roi_size, args.num_samples),
        )

        num_workers = min(args.num_workers, os.cpu_count() or 1)
        pin_memory = context.device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=list_data_collate,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )

        val_ds = Dataset(
            data=val_dicts,
            transform=dataset_config.val_transform_builder(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )

        try:
            model = build_model_with_filters(
                in_channels=dataset_config.in_channels,
                out_channels=dataset_config.out_channels,
                filters=args.model_filters,
            ).to(context.device)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "OOM while initializing DynUNet. Try smaller --roi-size "
                "(for example 96 96 96 or 64 64 64), keep --num-samples 1, "
                "and make sure Slurm launches exactly one task on one GPU."
            ) from exc

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
            "dataset_source": args.dataset_source,
            "data_dir": os.path.normpath(args.data_dir)
            if args.dataset_source == "brats"
            else "",
            "task_manifest": os.path.normpath(args.task_manifest)
            if args.dataset_source == "cancervision_binary_seg"
            else "",
            "in_channels": dataset_config.in_channels,
            "out_channels": dataset_config.out_channels,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_interval": args.val_interval,
            "seed": args.seed,
            "gpu_profile": gpu_profile.name,
            "roi_size": list(roi_size),
            "num_samples": args.num_samples,
            "model_filters": list(args.model_filters),
            "train_micro_batch_size": args.train_micro_batch_size,
            "test_size": args.test_size,
            "amp": args.amp,
            "val_sw_batch_size": args.val_sw_batch_size,
            "distributed": context.distributed,
            "world_size": context.world_size,
            "single_gpu_only": True,
        }
        wandb_run = maybe_init_wandb(args, context, wandb_config)

        for epoch in range(start_epoch, args.max_epochs):
            epoch_start = time.time()
            rank0_print(context, "-" * 40)
            rank0_print(context, f"Epoch {epoch + 1}/{args.max_epochs}")

            try:
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
                )
            except torch.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA OOM during training step. Reduce --roi-size, keep "
                    "--num-samples 1, and do not launch more than one Slurm task."
                ) from exc

            if is_main_process(context):
                print(f"  avg loss: {avg_loss:.4f}  lr: {current_lr:.2e}", flush=True)
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "epoch": epoch + 1,
                    }
                )

            if (epoch + 1) % args.val_interval == 0:
                try:
                    metrics = validate(
                        model=model,
                        val_loader=val_loader,
                        post_trans=post_trans,
                        roi_size=roi_size,
                        args=args,
                        context=context,
                        dataset_config=dataset_config,
                    )
                except torch.OutOfMemoryError as exc:
                    raise RuntimeError(
                        "CUDA OOM during validation. Lower --roi-size first; "
                        "--val-sw-batch-size is already safest at 1."
                    ) from exc
                if is_main_process(context) and metrics is not None:
                    if metrics["dice_mean"] > best_metric:
                        best_metric = metrics["dice_mean"]
                        best_metric_epoch = epoch + 1
                        model_state = model.state_dict()
                        torch.save(model_state, run_dir / "best_metric_model.pth")
                        print(
                            f"  -> saved new best model to {run_dir / 'best_metric_model.pth'}",
                            flush=True,
                        )

                    wandb_run.log(
                        {
                            "val/dice_mean": metrics["dice_mean"],
                            **{
                                f"val/dice_{metric_name}": metrics[f"dice_{metric_name}"]
                                for metric_name in dataset_config.metric_names
                            },
                            "val/best_dice": best_metric,
                            "epoch": epoch + 1,
                        }
                    )
                    metric_summary = " ".join(
                        f"{metric_name.upper()}={metrics[f'dice_{metric_name}']:.4f}"
                        for metric_name in dataset_config.metric_names
                    )
                    print(
                        f"  val dice: {metrics['dice_mean']:.4f}"
                        f"  ({metric_summary})"
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
