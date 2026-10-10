"""Pure-math loss kernels for refinement targets.

Each module holds one math function mirroring the tensor pipeline of the matching
``torchref.refinement.targets`` target, with all model/restraints/scaler/MixedTensor
bookkeeping stripped out. Signatures take only tensors and scalars -- this is the
boundary a Triton kernel replaces.

Re-exported here: ``bond``/``angle``/``chiral``/``planarity``/``torsion``/
``ramachandran``/``nonbonded`` for the geometry targets, ``adp.adp_simu_math``,
``xray_nll.nll_sigma_obs_math`` and ``xray_ls.ls_xray_loss_math``.

Everything else is imported from its module: the anisotropic-ADP family
(``adp.adp_*_aniso_math``, the U6 helpers) and the SIGD prior ``adp.adp_sigd_math``,
the model-error likelihoods and variance builders of ``xray_likelihoods``,
``xray_ml_full`` and ``dataset_scaling``. The likelihoods' ``beta`` comes from
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
