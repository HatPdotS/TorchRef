"""Optimized density-splatting kernels, organized by device.

* ``cpu/`` -- the production fused C++ spherical-cutoff splat (``sphere_splat.py``), the
  portable plain-scatter splats (``variable_radius.py``), and the JIT reference.
* ``cuda/`` -- the production variable-radius work-queue kernels
  (``variable_radius.py``) plus a fixed-radius fused Triton kernel (``fused.py``,
  benchmark-only).
* ``offsets.py`` -- shared voxel-offset helpers for the variable-radius splats.

The public API is re-exported here. Triton imports are guarded, so the package loads
without a GPU.
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

__all__ = [
    "vectorized_add_to_map",
    "build_electron_density",
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
# common real-world failure -- raises something other than ImportError on import. Catching
# only ImportError meant such a host could not import torchref at all, even though
# ``torchref.utils.triton_available()`` was written to absorb exactly this and would have
# reported False. The sibling guard in ``cuda/variable_radius.py`` already used the wider
# clause.
try:
    from .cuda.fused import fused_add_to_map_gpu

    # Appended rather than listed unconditionally. ``fused_add_to_map_gpu`` is bound only
    # if the import succeeded, so naming it in a static ``__all__`` made
    # ``from torchref.base.electron_density.kernels import *`` raise AttributeError on any
    # host without Triton.
    __all__.append("fused_add_to_map_gpu")
except Exception:  # pragma: no cover - depends on the host's triton install
    pass
