"""Correctness tests for the fused Dice CUDA kernel against MONAI DiceLoss.

Skipped when CUDA or nvcc is unavailable (e.g. CPU-only CI).
"""

from __future__ import annotations

import os
import shutil

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None or os.path.isfile("/usr/local/cuda/bin/nvcc")


pytestmark = pytest.mark.skipif(
    not (_cuda_available() and _nvcc_available()),
    reason="CUDA + nvcc required for fused kernel tests",
)


def _build(shape: tuple[int, ...], seed: int):
    import torch

    torch.manual_seed(seed)
    logits = torch.randn(*shape, device="cuda", requires_grad=True)
    targets = (torch.rand(*shape, device="cuda") > 0.5).float()
    return logits, targets


def test_forward_matches_monai():
    import torch
    from monai.losses import DiceLoss
    from src.kernels import FusedDiceLoss

    logits, targets = _build((2, 3, 16, 24, 32), seed=0)
    fused = FusedDiceLoss(smooth_nr=0.0, smooth_dr=1e-5)
    reference = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True, reduction="mean"
    )
    loss_f = fused(logits, targets)
    loss_r = reference(logits, targets)
    assert torch.allclose(loss_f, loss_r, atol=1e-4, rtol=1e-4)


def test_backward_matches_monai():
    import torch
    from monai.losses import DiceLoss
    from src.kernels import FusedDiceLoss

    logits, targets = _build((2, 3, 16, 24, 32), seed=1)
    fused = FusedDiceLoss(smooth_nr=0.0, smooth_dr=1e-5)
    reference = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True, reduction="mean"
    )

    loss_f = fused(logits, targets)
    loss_f.backward()
    grad_f = logits.grad.detach().clone()

    logits.grad = None
    loss_r = reference(logits, targets)
    loss_r.backward()
    grad_r = logits.grad.detach()

    assert torch.allclose(grad_f, grad_r, atol=1e-4, rtol=1e-3)


def test_handles_half_precision():
    import torch
    from monai.losses import DiceLoss
    from src.kernels import FusedDiceLoss

    torch.manual_seed(2)
    logits = torch.randn(1, 3, 8, 8, 8, device="cuda", dtype=torch.float16, requires_grad=True)
    targets = (torch.rand(1, 3, 8, 8, 8, device="cuda") > 0.5).to(torch.float16)

    fused = FusedDiceLoss(smooth_nr=0.0, smooth_dr=1e-5)
    reference = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True, reduction="mean"
    )
    loss_f = fused(logits, targets)
    loss_r = reference(logits.float(), targets.float())
    # fp16 accumulates in fp32 in both paths; allow a larger tolerance.
    assert torch.allclose(loss_f.float(), loss_r, atol=5e-3, rtol=5e-3)
