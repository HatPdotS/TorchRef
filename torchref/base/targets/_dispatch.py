"""Shared dispatch helper for the target math functions.

When called on a CUDA float32 tensor and Triton is importable, the math
functions in this package transparently route to their Triton-kernel
implementations in :mod:`torchref.base.targets.triton`. CPU tensors,
non-float32 tensors, or environments without Triton fall back to the
plain eager implementation.

Single import gate (cheap) — no per-call try/except.
"""

import torch


try:
    import triton  # noqa: F401
    HAS_TRITON: bool = True
except ImportError:
    HAS_TRITON = False


def use_triton(*tensors: torch.Tensor) -> bool:
    """Decide whether to route a call to the Triton kernel.

    Requires Triton importable AND every probed tensor to be CUDA
    float32 (the only configuration the kernels are written for).
    """
    if not HAS_TRITON:
        return False
    for t in tensors:
        if t is None:
            continue
        if not t.is_cuda:
            return False
        if t.dtype != torch.float32:
            return False
    return True
