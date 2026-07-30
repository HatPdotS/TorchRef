"""
Backward-compatibility shim.

The density-splatting kernels moved to
:mod:`torchref.base.electron_density.kernels` (organized into ``cpu``/``cuda``/
``mps`` subpackages). This module re-exports the public API from the new location
so existing ``from torchref.base.kernels import ...`` imports keep working.
Prefer importing from ``torchref.base.electron_density.kernels`` in new code.
"""

from torchref.base.electron_density.kernels import (  # noqa: F401
    vectorized_add_to_map,
    build_electron_density,
    compute_metric_tensor,
    precompute_fractional_coords,
    warmup,
    get_cache_dir,
    clear_cache,
)

__all__ = [
    "vectorized_add_to_map",
    "build_electron_density",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
]

# Optional, and only advertised when it actually resolved -- listing it
# unconditionally made ``from torchref.base.kernels import *`` raise on a Triton-less
# host. The former ``_HAS_TRITON`` re-export is gone: nothing read it, and
# ``torchref.utils.triton_available()`` is the answer to that question.
try:
    from torchref.base.electron_density.kernels import (  # noqa: F401
        fused_add_to_map_gpu,
    )

    __all__.append("fused_add_to_map_gpu")
except ImportError:
    pass
