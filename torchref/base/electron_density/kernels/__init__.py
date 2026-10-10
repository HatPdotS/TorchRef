"""Density-splatting kernels, organized by device.

* ``cpu/`` -- ``sphere_splat`` (the production fused CPU splat, prebuilt in ``torchref-kernels``), ``variable_radius``
  (the portable base case) and ``jit_reference`` (the voxel-list API).
* ``cuda/`` -- ``variable_radius`` (the production splat) and ``fused`` (the Triton
  branch of the voxel-list API). ``mps/`` -- the Metal splat.

The re-exports are the voxel-list API; production density goes through
``electron_density.main.build_electron_density``. Triton imports are guarded, so the
package loads without a GPU.
"""

from .cpu.jit_reference import (
    vectorized_add_to_map,
    compute_metric_tensor,
    precompute_fractional_coords,
    warmup,
    get_cache_dir,
    clear_cache,
)

__all__ = [
    "vectorized_add_to_map",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
]

# Triton kernels are optional (they require the triton package and a GPU).
#
# ``except Exception``, not ``except ImportError``: this runs during ``import torchref``,
# and a Triton install that is present but broken -- a driver or LLVM version skew, the
# common real-world failure -- raises something other than ImportError on import.
try:
    from .cuda.fused import fused_add_to_map_gpu

    # Appended only when bound, so ``import *`` cannot raise AttributeError without Triton.
    __all__.append("fused_add_to_map_gpu")
except Exception:  # pragma: no cover - depends on the host's triton install
    pass
