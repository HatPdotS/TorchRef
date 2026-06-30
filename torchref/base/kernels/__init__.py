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
    _HAS_TRITON,
    _HAS_SEPARABLE_TRITON,
)

try:
    from torchref.base.electron_density.kernels import (  # noqa: F401
        fused_add_to_map_gpu,
        separable_density_gpu,
    )
except ImportError:
    pass

__all__ = [
    "vectorized_add_to_map",
    "build_electron_density",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
    "fused_add_to_map_gpu",
    "separable_density_gpu",
]
