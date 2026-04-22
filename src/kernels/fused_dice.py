"""Custom CUDA fused Dice loss.

Wraps two hand-written CUDA kernels (`fused_dice.cu`) with a torch.autograd
Function and a tiny nn.Module surface. On first use the extension is JIT
compiled via torch.utils.cpp_extension.load_inline and cached under
``~/.cache/torch_extensions`` (or ``$TORCH_EXTENSIONS_DIR``).

Numerically equivalent to:
    monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-5,
        reduction="mean", to_onehot_y=False,
    )

Why write this ourselves
------------------------
The MONAI implementation (and most PyTorch Dice variants) issue a sequence of
discrete ops: sigmoid -> multiply -> reduce(intersection) -> square -> reduce
(cardinality) -> compose -> mean. That is 6+ kernel launches, each reading the
large (B, C, D, H, W) activation tensor from HBM. For 3D BraTS volumes this
tensor is tens of megabytes per supervision head; the whole computation is
memory bandwidth bound.

The fused kernel reads each logit/target pair *once* per forward and once per
backward, accumulating intersection and cardinality inline in registers, then
does a single shared-memory reduction per block. That is 1 launch vs ~6, and
1x tensor read vs ~4x.

Expected speedup: ~2-3x for the loss ops themselves; ~5-10% overall step time
on A100 at 192**3 ROI with deep supervision enabled (loss is called 4x per
step). Verified end-to-end against MONAI in the preflight step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

_CPP_SOURCE = r"""
#include <torch/extension.h>

void fused_dice_forward_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor);
void fused_dice_backward_cuda(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    double, double, double
);

void fused_dice_forward(
    at::Tensor logits, at::Tensor targets,
    at::Tensor intersection, at::Tensor cardinality
) {
    TORCH_CHECK(logits.is_cuda() && targets.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(logits.is_contiguous() && targets.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(logits.sizes() == targets.sizes(), "shape mismatch");
    TORCH_CHECK(intersection.scalar_type() == at::kFloat, "intersection must be float32");
    TORCH_CHECK(cardinality.scalar_type() == at::kFloat, "cardinality must be float32");
    fused_dice_forward_cuda(logits, targets, intersection, cardinality);
}

void fused_dice_backward(
    at::Tensor logits, at::Tensor targets,
    at::Tensor intersection, at::Tensor cardinality,
    at::Tensor grad_logits,
    double eps_n, double eps_d, double grad_scale
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(grad_logits.sizes() == logits.sizes(), "grad shape mismatch");
    fused_dice_backward_cuda(
        logits, targets, intersection, cardinality, grad_logits,
        eps_n, eps_d, grad_scale
    );
}
"""


_module: Optional[object] = None


def _get_module():
    global _module
    if _module is not None:
        return _module

    cuda_source = (Path(__file__).with_suffix(".cu")).read_text()
    build_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    kwargs = {
        "name": "cancervision_fused_dice",
        "cpp_sources": [_CPP_SOURCE],
        "cuda_sources": [cuda_source],
        "functions": ["fused_dice_forward", "fused_dice_backward"],
        "extra_cuda_cflags": ["-O3", "--use_fast_math"],
        "extra_cflags": ["-O3"],
        "verbose": False,
    }
    if build_dir:
        kwargs["build_directory"] = build_dir
    _module = load_inline(**kwargs)
    return _module


class _FusedDiceFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, eps_n: float, eps_d: float):
        if logits.dim() < 2:
            raise ValueError(f"logits must have >=2 dims (B, C, ...); got {logits.shape}")
        if logits.shape != targets.shape:
            raise ValueError(f"shape mismatch: logits {logits.shape} vs targets {targets.shape}")

        logits_c = logits.contiguous()
        targets_c = targets.to(dtype=logits_c.dtype).contiguous()
        B, C = logits_c.shape[0], logits_c.shape[1]
        intersection = torch.zeros(B * C, dtype=torch.float32, device=logits_c.device)
        cardinality = torch.zeros(B * C, dtype=torch.float32, device=logits_c.device)

        module = _get_module()
        module.fused_dice_forward(logits_c, targets_c, intersection, cardinality)

        # Epilogue in float32: (1 - (2I + e_n) / (C + e_d)).mean()
        per_bc = 1.0 - (2.0 * intersection + eps_n) / (cardinality + eps_d)
        loss = per_bc.mean()

        ctx.save_for_backward(logits_c, targets_c, intersection, cardinality)
        ctx.eps_n = float(eps_n)
        ctx.eps_d = float(eps_d)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, targets, intersection, cardinality = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        # grad_output is a scalar; syncing once per backward is negligible.
        grad_scale = float(grad_output.detach().to(torch.float64).item())
        module = _get_module()
        module.fused_dice_backward(
            logits, targets, intersection, cardinality, grad_logits,
            ctx.eps_n, ctx.eps_d, grad_scale,
        )
        return grad_logits, None, None, None


class FusedDiceLoss(torch.nn.Module):
    """Fused sigmoid + squared-predicted Dice loss (CUDA only).

    Args:
        smooth_nr: epsilon added to 2*intersection  (MONAI smooth_nr)
        smooth_dr: epsilon added to cardinality     (MONAI smooth_dr)
    """

    def __init__(self, smooth_nr: float = 0.0, smooth_dr: float = 1e-5):
        super().__init__()
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _FusedDiceFn.apply(logits, targets, self.smooth_nr, self.smooth_dr)
