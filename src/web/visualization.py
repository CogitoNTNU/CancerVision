"""Visualization helpers for web inference outputs."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def create_preview_png(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    output_path: str,
    title: str,
) -> None:
    """Create side-by-side preview PNG from volume center slice."""
    if image.ndim != 4:
        raise ValueError("Expected image tensor shape (C, H, W, D)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image_np = image.cpu().numpy()
    flair = image_np[0]
    z_index = flair.shape[2] // 2
    base_slice = flair[:, :, z_index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
    axes[0].imshow(base_slice, cmap="gray")
    axes[0].set_title("Input (center slice)")
    axes[0].axis("off")

    axes[1].imshow(base_slice, cmap="gray")
    if mask is not None:
        if mask.ndim != 4:
            raise ValueError("Expected mask tensor shape (C, H, W, D)")
        mask_np = mask.cpu().numpy()
        label_map = np.zeros(mask_np.shape[1:], dtype=np.uint8)
        tc = mask_np[0] > 0
        wt = mask_np[1] > 0
        et = mask_np[2] > 0
        label_map[wt] = 1
        label_map[tc] = 2
        label_map[et] = 3

        overlay = label_map[:, :, z_index]
        axes[1].imshow(overlay, cmap="jet", alpha=0.45, vmin=0, vmax=3)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
