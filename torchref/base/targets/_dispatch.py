"""Shared dispatch helper for the target math functions.

When called on a CUDA float32 tensor and Triton is importable, the math
functions in this package transparently route to their Triton-kernel
implementations in :mod:`torchref.base.targets.triton`. CPU tensors,
non-float32 tensors, or environments without Triton fall back to the
plain eager implementation.

Selection is governed by the shared, capability-based ``Engine`` in
:mod:`torchref.utils.triton_dispatch` (no environment variables). To force
the eager path for an A/B comparison or to sidestep a flaky Triton install::

    from torchref.utils import use_engine, Engine
    with use_engine(Engine.EAGER):
        ...
"""

import torch

from torchref.utils.triton_dispatch import should_use_triton


def use_triton(*tensors: torch.Tensor) -> bool:
    """Decide whether to route a call to the Triton kernel.

    Thin wrapper over :func:`torchref.utils.triton_dispatch.should_use_triton`
    using the process-wide engine: Triton is used only when the engine permits
    it and every probed tensor is CUDA float32 (the only configuration the
    kernels are written for).
    """
    return should_use_triton(*tensors)
