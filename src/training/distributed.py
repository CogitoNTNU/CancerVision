"""Device and distributed-launch setup for training.

Supports three launch modes:

    * Single process on CPU / MPS / single CUDA device.
    * torchrun with WORLD_SIZE/RANK/LOCAL_RANK env vars.
    * Slurm with SLURM_NTASKS/SLURM_PROCID/SLURM_LOCALID (no torchrun needed).

The caller drives setup via `setup_runtime()` and teardown via `cleanup()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    device_index: int | None
    distributed: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value or not value.lstrip("-").isdigit():
        return None
    return int(value)


def _count_csv(value: str | None) -> int | None:
    if value is None:
        return None
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    return len(entries) if entries else None


def _detect_allocated_gpus(env: Mapping[str, str]) -> int | None:
    for resolver in (
        lambda: _parse_int(env.get("SLURM_GPUS_ON_NODE")),
        lambda: _count_csv(env.get("SLURM_STEP_GPUS")),
        lambda: _count_csv(env.get("CUDA_VISIBLE_DEVICES")),
    ):
        value = resolver()
        if value is not None:
            return value
    return None


def _resolve_ranks(env: Mapping[str, str]) -> tuple[int, int, int]:
    world_size = int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", "1")))
    rank = int(env.get("RANK", env.get("SLURM_PROCID", "0")))
    local_rank = int(env.get("LOCAL_RANK", env.get("SLURM_LOCALID", "0")))
    return rank, local_rank, world_size


def _pick_device_index(
    local_rank: int,
    visible_gpus: int,
    allocated_gpus: int | None,
    world_size: int,
) -> int:
    if visible_gpus < 1:
        raise RuntimeError("Distributed training requires at least one visible GPU")
    if allocated_gpus is not None and allocated_gpus < world_size:
        raise RuntimeError(
            f"Allocated GPUs ({allocated_gpus}) < world size ({world_size})"
        )
    if visible_gpus == 1:
        return 0
    if 0 <= local_rank < visible_gpus:
        return local_rank
    raise RuntimeError(
        f"Cannot map local_rank={local_rank} to visible_gpus={visible_gpus}"
    )


def setup_runtime() -> RuntimeContext:
    env = os.environ
    rank, local_rank, world_size = _resolve_ranks(env)
    distributed = world_size > 1
    cuda_available = torch.cuda.is_available()
    visible_gpus = torch.cuda.device_count() if cuda_available else 0

    if distributed:
        if not cuda_available:
            raise RuntimeError("Distributed training requires CUDA")
        device_index = _pick_device_index(
            local_rank, visible_gpus, _detect_allocated_gpus(env), world_size
        )
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")
        env.setdefault("WORLD_SIZE", str(world_size))
        env.setdefault("RANK", str(rank))
        env.setdefault("LOCAL_RANK", str(local_rank))
        torch.cuda.set_device(device_index)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        return RuntimeContext(
            device=torch.device("cuda", device_index),
            device_index=device_index,
            distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

    if cuda_available:
        return RuntimeContext(
            device=torch.device("cuda:0"),
            device_index=0,
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return RuntimeContext(
            device=torch.device("mps"),
            device_index=None,
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
        )
    return RuntimeContext(
        device=torch.device("cpu"),
        device_index=None,
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
    )


def cleanup(context: RuntimeContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: RuntimeContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, count: int, context: RuntimeContext) -> float:
    if not context.distributed:
        return value / max(count, 1)
    tensor = torch.tensor([value, float(count)], device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor[0] / tensor[1].clamp_min(1.0)).item()


def rank0_print(context: RuntimeContext, *args, **kwargs) -> None:
    if context.is_main:
        print(*args, flush=True, **kwargs)
