"""Custom CUDA kernels for the CancerVision training pipeline.

Kernels are JIT compiled via torch.utils.cpp_extension.load_inline on first
use and cached under ``~/.cache/torch_extensions``. Requires nvcc + the CUDA
toolkit on the host.
"""

from .fused_dice import FusedDiceLoss

__all__ = ["FusedDiceLoss"]
