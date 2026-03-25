"""Dataset-agnostic preprocessing helpers."""

from __future__ import annotations

import torch


def sanitize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """Return float32 tensor with NaN/Inf replaced by finite values."""
    if image.ndim != 4:
        raise ValueError("Expected image tensor shape (C, H, W, D)")

    sanitized = image.clone().float()
    sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
    return sanitized


def clip_percentile_nonzero_per_channel(
    image: torch.Tensor,
    lower: float = 0.5,
    upper: float = 99.5,
) -> torch.Tensor:
    """Clip each channel to non-zero voxel percentiles."""
    if not (0.0 <= lower < upper <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    clipped = image.clone()
    for channel in range(clipped.shape[0]):
        voxels = clipped[channel]
        nonzero = voxels[voxels != 0]
        if nonzero.numel() == 0:
            continue

        lo = torch.quantile(nonzero, lower / 100.0)
        hi = torch.quantile(nonzero, upper / 100.0)
        clipped[channel] = voxels.clamp(min=lo.item(), max=hi.item())

    return clipped


def zscore_nonzero_per_channel(image: torch.Tensor) -> torch.Tensor:
    """Apply per-channel z-score normalization over non-zero voxels."""
    if image.ndim != 4:
        raise ValueError("Expected image tensor shape (C, H, W, D)")

    normalized = image.clone().float()
    for channel in range(normalized.shape[0]):
        voxels = normalized[channel]
        mask = voxels != 0
        if mask.any():
            values = voxels[mask]
            voxels[mask] = (values - values.mean()) / (values.std() + 1e-8)
            normalized[channel] = voxels
    return normalized


def preprocess_image_volume(
    image: torch.Tensor,
    clip_lower_percentile: float = 0.5,
    clip_upper_percentile: float = 99.5,
) -> torch.Tensor:
    """Apply robust preprocessing for model consumption.

    Order:
    1. Sanitize NaN/Inf values.
    2. Clip each channel to non-zero percentiles.
    3. Apply non-zero z-score normalization per channel.
    """
    sanitized = sanitize_image_tensor(image)
    clipped = clip_percentile_nonzero_per_channel(
        sanitized,
        lower=clip_lower_percentile,
        upper=clip_upper_percentile,
    )
    normalized = zscore_nonzero_per_channel(clipped)
    return normalized
