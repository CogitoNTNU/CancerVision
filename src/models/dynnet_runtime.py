"""Runtime helpers for the single-GPU DynUNet trainer."""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
from dataclasses import dataclass
from typing import Mapping

import torch

from src.models.dynnet_config import (
    DEFAULT_GPU_PROFILE_NAME,
    GPU_PROFILE_ALIASES,
    GPU_PROFILE_CONFIGS,
    GpuProfileConfig,
)


@dataclass
class RuntimeContext:
    device: torch.device
    device_index: int | None
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


def get_env_int(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    value = env.get(name)
    if value in (None, ""):
        return default
    return int(value)


def bootstrap_distributed_env(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    target_env = os.environ if env is None else env

    if "SLURM_PROCID" not in target_env:
        return target_env

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

    return target_env


def get_distributed_env(env: Mapping[str, str] | None = None) -> tuple[int, int, int]:
    env = bootstrap_distributed_env(env)
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
    if visible_device_count < 1:
        raise RuntimeError("CUDA is available but no visible devices were reported")
    if visible_device_count == 1:
        return 0
    if local_rank < 0 or local_rank >= visible_device_count:
        raise RuntimeError(
            f"Invalid local rank {local_rank} for {visible_device_count} visible CUDA devices"
        )
    return local_rank


def is_main_process(context: RuntimeContext) -> bool:
    return context.rank == 0


def should_use_amp(args: argparse.Namespace, context: RuntimeContext) -> bool:
    return bool(args.amp and context.device.type == "cuda")


def _bytes_to_gib(total_bytes: int) -> float:
    return total_bytes / (1024**3)


def format_cuda_memory_summary(context: RuntimeContext) -> str | None:
    if context.device.type != "cuda":
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(context.device)
    except Exception:
        return None
    return f"{_bytes_to_gib(free_bytes):.1f} GiB free / {_bytes_to_gib(total_bytes):.1f} GiB total"


def is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True

    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        lowered = str(exc).lower()
        return "out of memory" in lowered or "cudaerrormemoryallocation" in lowered

    if isinstance(exc, RuntimeError):
        lowered = str(exc).lower()
        return "cuda" in lowered and (
            "out of memory" in lowered or "cudaerrormemoryallocation" in lowered
        )

    return False


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

    total_memory_gib = _bytes_to_gib(
        torch.cuda.get_device_properties(context.device).total_memory
    )
    available_memory_gib = total_memory_gib
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(context.device)
        available_memory_gib = min(available_memory_gib, _bytes_to_gib(free_bytes))
    except Exception:
        pass

    if available_memory_gib <= 20:
        return "gpu16g"
    if available_memory_gib <= 36:
        return "gpu32g"
    if available_memory_gib <= 48:
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
        device_index = resolve_cuda_device_index(local_rank, torch.cuda.device_count())
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
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


def get_autocast_context(
    context: RuntimeContext,
    enabled: bool,
) -> contextlib.AbstractContextManager:
    if enabled and context.device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def build_micro_batch_slices(
    total_batch_size: int,
    requested_micro_batch_size: int,
) -> list[slice]:
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
