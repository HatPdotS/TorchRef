"""
Optimized kernels for electron density computation.

This submodule provides optimized implementations for compute-intensive
operations in crystallographic calculations:

- JIT-compiled PyTorch kernels for CPU and GPU
- Triton CUDA kernels for fused operations
- Optimized fused operations with reduced kernel launches
"""

from .jit_kernel_vectorized_add_to_map import (
    vectorized_add_to_map,
    build_electron_density,
    compute_metric_tensor,
    precompute_fractional_coords,
    warmup,
    get_cache_dir,
    clear_cache,
)

# Triton kernels are optional (require triton package)
try:
    from .triton_kernel import fused_add_to_map_gpu, fused_find_and_place_atoms
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

try:
    from .separable_triton_kernel import separable_density_gpu
    _HAS_SEPARABLE_TRITON = True
except ImportError:
    _HAS_SEPARABLE_TRITON = False

__all__ = [
    # JIT kernels
    "vectorized_add_to_map",
    "build_electron_density",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
    # Triton kernels (if available)
    "fused_add_to_map_gpu",
    "fused_find_and_place_atoms",
    "separable_density_gpu",
]
