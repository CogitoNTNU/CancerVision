"""Configuration and CLI helpers for the DynUNet trainer."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from dotenv import load_dotenv
from monai.transforms import Compose

from src.datasets import default_brats_data_dir

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def validate_args(args: argparse.Namespace) -> None:
    if args.train_micro_batch_size < 1:
        raise ValueError("--train-micro-batch-size must be at least 1")
    if args.val_sw_batch_size < 1:
        raise ValueError("--val-sw-batch-size must be at least 1")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if len(args.roi_size) != 3 or any(size < 32 for size in args.roi_size):
        raise ValueError("--roi-size must contain three integers >= 32")
    if len(args.model_filters) != 5 or any(width < 8 for width in args.model_filters):
        raise ValueError("--model-filters must contain five integers >= 8")
