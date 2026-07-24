"""Metal (MPS) variable-radius electron-density splat kernels.

Native Metal kernels (compiled at runtime via ``torch.mps.compile_shader``) for
the density splat on Apple-silicon GPUs, replacing the portable eager splat that
dominates fcalc time on MPS. Gated behind ``device.type == 'mps'`` in
``electron_density.main``; every other platform is unaffected.
"""

from torchref.base.electron_density.kernels.mps.compile import (
    clear_cache,
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
]
