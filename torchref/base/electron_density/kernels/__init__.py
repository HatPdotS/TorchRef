"""
Optimized density-splatting kernels, organized by device.

Layout:
- ``cpu/``  — the per-atom variable-radius CPU splats
  (``variable_radius.py``: the production CPU AUTO grouped-separable and the
  portable plain-scatter splats), the shared separable density core
  (``separable.py``), the aniso splat, the C++ parallel scatter
  (``scatter.py`` / ``scatter_dispatch.py``), and the JIT reference.
- ``cuda/`` — the production variable-radius work-queue kernels
  (``variable_radius.py``: ``WorkQueueGridDensity{,Aniso}``) plus the legacy
  fixed-radius fused Triton kernel (``fused.py``, benchmark-only).
- ``offsets.py`` — shared voxel-offset helpers for the variable-radius splats.

This package re-exports the public API (``vectorized_add_to_map``, the two-step
``build_electron_density``, the variable-radius entry points, and the legacy
Triton entry points, …). Triton imports are optional (guarded) so the package
loads without a GPU.
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

__all__ = [
    "vectorized_add_to_map",
    "build_electron_density",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
    "fused_add_to_map_gpu",
]
