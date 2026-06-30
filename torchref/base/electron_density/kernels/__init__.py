"""
Optimized density-splatting kernels, organized by device.

Layout:
- ``cpu/``  — CPU separable/fused/aniso splats, the C++ parallel scatter, the
  scatter dispatcher, the eager two-step reference, and the JIT reference.
- ``cuda/`` — fused + separable Triton kernels.
- ``mps/``  — MPS single-pass splat.
- ``offsets.py`` — shared voxel-offset helpers (used by cpu + mps).

This package re-exports the public API (``vectorized_add_to_map``, the two-step
``build_electron_density``, the Triton entry points, …). Triton imports are
optional (guarded) so the package loads without a GPU.
"""

from .cpu.jit_reference import (
    vectorized_add_to_map,
    build_electron_density,
    compute_metric_tensor,
    precompute_fractional_coords,
    warmup,
    get_cache_dir,
    clear_cache,
)

# Triton kernels are optional (require the triton package / CUDA).
try:
    from .cuda.fused import fused_add_to_map_gpu
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

try:
    from .cuda.separable import separable_density_gpu
    _HAS_SEPARABLE_TRITON = True
except ImportError:
    _HAS_SEPARABLE_TRITON = False

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
