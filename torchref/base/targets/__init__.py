"""Pure-math loss kernels for refinement targets.

Each module in this package provides a single math function that mirrors the
tensor pipeline inside the corresponding ``torchref.refinement.targets``
target, with all model/restraints/scaler/MixedTensor bookkeeping stripped
out. The signatures take only tensors and scalars — this is the boundary a
Triton kernel would replace.

Mapping
-------

    refinement.targets.geometry.bonds.BondTarget               -> bond.bond_math
    refinement.targets.geometry.angles.AngleTarget             -> angle.angle_math
    refinement.targets.geometry.chiral.ChiralTarget            -> chiral.chiral_math
    refinement.targets.geometry.planarity.PlanarityTarget      -> planarity.planarity_math
    refinement.targets.geometry.torsions.TorsionTarget         -> torsion.torsion_omega_math
    refinement.targets.geometry.ramachandran.RamachandranTarget -> ramachandran.ramachandran_math
    refinement.targets.geometry.non_bonded.NonBondedTarget     -> nonbonded.nonbonded_heavy_math
    refinement.targets.adp.similarity.ADPSimilarityTarget      -> adp.adp_simu_math
    refinement.targets.xray.rice.RiceXrayTarget                -> xray_ml.ml_xray_loss_math
    refinement.targets.xray.gaussian.GaussianXrayTarget        -> xray_gaussian.gaussian_xray_loss_math
    refinement.targets.xray.least_squares.LeastSquaresXrayTarget -> xray_ls.ls_xray_loss_math
    refinement.targets.xray.bhattacharyya.BhattacharyyaXrayTarget -> xray_bhattacharyya.bhattacharyya_xray_loss_math

The anisotropic-ADP restraint family and the renamed ``ml`` x-ray mode live in
this package but are not re-exported above:

    adp.adp_simu_aniso_math, adp.adp_locality_aniso_math,
    adp.adp_rigid_bond_aniso_math      (aniso counterparts of adp_simu_math)
    xray_ml_sigmaa.ml_xray_loss_beta_math
                                       (the math used by ``mode="ml"``; the
                                        ``ml_sigmaa`` name is a legacy alias)
"""

from .adp import adp_simu_math
from .angle import angle_math
from .bond import bond_math
from .chiral import chiral_math
from .nonbonded import nonbonded_heavy_math
from .planarity import planarity_math
from .ramachandran import ramachandran_math
from .torsion import torsion_omega_math
from .xray_bhattacharyya import bhattacharyya_xray_loss_math
from .xray_gaussian import gaussian_xray_loss_math
from .xray_ls import ls_xray_loss_math
from .xray_ml import ml_xray_loss_math

__all__ = [
    "adp_simu_math",
    "angle_math",
    "bhattacharyya_xray_loss_math",
    "bond_math",
    "chiral_math",
    "gaussian_xray_loss_math",
    "ls_xray_loss_math",
    "ml_xray_loss_math",
    "nonbonded_heavy_math",
    "planarity_math",
    "ramachandran_math",
    "torsion_omega_math",
]
