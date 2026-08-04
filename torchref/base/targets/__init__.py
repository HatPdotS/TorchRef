"""Pure-math loss kernels for refinement targets.

Each module holds one math function mirroring the tensor pipeline of the matching
``torchref.refinement.targets`` target, with all model/restraints/scaler/MixedTensor
bookkeeping stripped out. Signatures take only tensors and scalars -- this is the
boundary a Triton kernel replaces.

Re-exported here: ``bond``/``angle``/``chiral``/``planarity``/``torsion``/
``ramachandran``/``nonbonded`` for the geometry targets, ``adp.adp_simu_math``,
``xray_nll.nll_sigma_obs_math`` and ``xray_ls.ls_xray_loss_math``.

Deliberately **not** re-exported, though they live here: the anisotropic-ADP family
(``adp.adp_*_aniso_math``) and the model-error likelihoods
(``xray_likelihoods.rice_math`` / ``nll_math`` / ``rice_marginal_math`` and the
variance builders). They are the heaviest modules in the package and importing them
eagerly would drag them into every consumer. Their ``beta`` comes from
:mod:`torchref.refinement.model_error_estimation`, not from here.
"""

from .adp import adp_simu_math
from .angle import angle_math
from .bond import bond_math
from .chiral import chiral_math
from .nonbonded import nonbonded_heavy_math
from .planarity import planarity_math
from .ramachandran import ramachandran_math
from .torsion import torsion_omega_math
from .xray_nll import nll_sigma_obs_math
from .xray_ls import ls_xray_loss_math

__all__ = [
    "adp_simu_math",
    "angle_math",
    "bond_math",
    "chiral_math",
    "nll_sigma_obs_math",
    "ls_xray_loss_math",
    "nonbonded_heavy_math",
    "planarity_math",
    "ramachandran_math",
    "torsion_omega_math",
]
