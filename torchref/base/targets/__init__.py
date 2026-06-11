"""Pure-math loss kernels for refinement targets.

Each module in this package provides a single math function that mirrors the
tensor pipeline inside the corresponding ``torchref.refinement.targets``
target, with all model/restraints/scaler/MixedTensor bookkeeping stripped
out. The signatures take only tensors and scalars — this is the boundary a
Triton kernel would replace.

Mapping
-------

    refinement.targets.geometry.BondTarget         -> bond.bond_math
    refinement.targets.geometry.AngleTarget        -> angle.angle_math
    refinement.targets.geometry.ChiralTarget       -> chiral.chiral_math
    refinement.targets.geometry.PlanarityTarget    -> planarity.planarity_math
    refinement.targets.geometry.TorsionTarget      -> torsion.torsion_omega_math
    refinement.targets.geometry.RamachandranTarget -> ramachandran.ramachandran_math
    refinement.targets.geometry.NonBondedTarget    -> nonbonded.nonbonded_heavy_math
    refinement.targets.adp.ADPSimilarityTarget     -> adp.adp_simu_math
    refinement.targets.xray.RiceXrayTarget          -> xray_ml.ml_xray_loss_math
    refinement.targets.xray.GaussianXrayTarget          -> xray_gaussian.gaussian_xray_loss_math
    refinement.targets.xray.LeastSquaresXrayTarget      -> xray_ls.ls_xray_loss_math
    refinement.targets.xray.BhattacharyyaXrayTarget     -> xray_bhattacharyya.bhattacharyya_xray_loss_math
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
