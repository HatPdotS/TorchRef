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

from .optimized_ops import (
    fused_gaussian_density,
    fused_aniso_gaussian_density,
    warmup_cuda_operations,
    compute_smallest_diff_squared,
    CachedRadiusMask,
    get_cached_radius_offsets,
    vectorized_add_to_map_optimized,
)

# Triton kernel is optional (requires triton package)
try:
    from .triton_kernel import fused_add_to_map_gpu, fused_find_and_place_atoms
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

__all__ = [
    # JIT kernels
    "vectorized_add_to_map",
    "build_electron_density",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
    # Optimized ops
    "fused_gaussian_density",
    "fused_aniso_gaussian_density",
    "warmup_cuda_operations",
    "compute_smallest_diff_squared",
    "CachedRadiusMask",
    "get_cached_radius_offsets",
    "vectorized_add_to_map_optimized",
    # Triton kernel (if available)
    "fused_add_to_map_gpu",
    "fused_find_and_place_atoms",
]
