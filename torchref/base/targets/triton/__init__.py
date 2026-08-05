"""Triton kernels for geometry-target math.

Each module in this package provides a Triton-backed implementation of the
corresponding eager-mode function in :mod:`torchref.base.targets`.

Naming convention: ``<target>_math_triton``. The eager versions remain in
the parent package as reference implementations and equivalence baselines.

Two additional public kernels are intentionally not re-exported here, as they
are not part of the dispatched ``<target>_math_triton`` API and are invoked
directly by their callers: ``torsion.torsion_unimodal_full_math_triton`` (the
unimodal-torsion path) and ``place_hydrogens.place_riding_hydrogens_triton``
(riding-H placement).
"""

from .adp_simu import adp_simu_math_triton
from .angle import angle_math_triton
from .bond import bond_math_triton
from .chiral import chiral_math_triton
from .nonbonded import nonbonded_heavy_math_triton
from .planarity import planarity_math_triton
from .ramachandran import ramachandran_math_triton
from .torsion import torsion_omega_math_triton
from .xray_nll import nll_sigma_obs_math_triton
from .xray_ls import ls_xray_loss_math_triton

__all__ = [
    "adp_simu_math_triton",
    "angle_math_triton",
    "bond_math_triton",
    "chiral_math_triton",
    "nll_sigma_obs_math_triton",
    "ls_xray_loss_math_triton",
    "nonbonded_heavy_math_triton",
    "planarity_math_triton",
    "ramachandran_math_triton",
    "torsion_omega_math_triton",
]
