"""Metal (MPS) variable-radius electron-density splat kernels.

Native Metal kernels (compiled at runtime via ``torch.mps.compile_shader``) for
the density splat on Apple-silicon GPUs, replacing the portable eager splat that
dominates fcalc time on MPS. Selection goes through
``torchref.utils.should_use_metal``, which gates on MPS + float32 + a compiled
shader; a runtime failure degrades to the portable splat and warns.
Every other platform is unaffected.
"""

from torchref.base.electron_density.kernels.mps.compile import (
    clear_cache,
    last_error,
    mps_kernels_available,
    warmup,
)
from torchref.base.electron_density.kernels.mps.variable_radius import (
    add_anisotropic_mps_var,
    add_isotropic_mps_var,
)

__all__ = [
    "add_isotropic_mps_var",
    "add_anisotropic_mps_var",
    "mps_kernels_available",
    "warmup",
    "clear_cache",
    "last_error",
]
