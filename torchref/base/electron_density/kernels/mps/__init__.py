"""Metal (MPS) variable-radius electron-density splat kernels.

Native Metal kernels (compiled at runtime via ``torch.mps.compile_shader``) for
the density splat on Apple-silicon GPUs, replacing the portable eager splat that
dominates fcalc time on MPS. Selection is the ``mps_metal`` row of
``_backends.DENSITY_BACKENDS`` (MPS + float32, probed by ``compile.why_unavailable``):
an unavailable shader falls through to the portable splat silently, while a runtime
failure degrades with a ``TorchRefDegradationWarning``.
Every other platform is unaffected.
"""

from torchref.base.electron_density.kernels.mps.compile import last_error
from torchref.base.electron_density.kernels.mps.variable_radius import (
    add_anisotropic_mps_var,
    add_isotropic_mps_var,
)

__all__ = [
    "add_isotropic_mps_var",
    "add_anisotropic_mps_var",
    "last_error",
]
