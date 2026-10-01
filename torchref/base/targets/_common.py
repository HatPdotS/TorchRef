"""Shared helpers for target math kernels.

Also home to :func:`torsions_from_xyz`, the package's only eager dihedral, which the
topology layer uses as well; :mod:`torchref.base.targets.triton._dihedral` is its Triton
counterpart.
"""

import numpy as np
import torch

LOG_2PI: float = float(np.log(2.0 * np.pi))
DEG2RAD: float = float(np.pi) / 180.0
RAD2DEG: float = 180.0 / float(np.pi)

# Safe-divide floor for the eager geometry math. Mirrors the guards in the
# Triton kernels so the eager path produces FINITE gradients at degenerate
# geometry (zero-length bonds, collinear angles/torsions) instead of NaN.
# 1e-6 is small enough to leave non-degenerate values unchanged yet large
# enough to be representable in float32.
EPS: float = 1e-6
# Clamp bound for cosines fed to ``acos``: keeps ``1 - cos**2`` (and hence
# ``acos``'s ``-1/sqrt(1-cos**2)`` backward) finite at exact collinearity.
COS_CLAMP: float = 1.0 - EPS


def torsions_from_xyz(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Compute dihedral angles in degrees from 4-atom indices.

    The one eager dihedral in the package: ``Restraints.torsions``, the omega that
    classifies cis/trans proline for the Ramachandran map and every eager geometry
    target call it.

    The sign is IUPAC, the same as ``gemmi.calculate_dihedral`` and the convention the
    CCP4/AceDRG monomer-library references are written in: for atoms A-B-C-D viewed
    along the B→C bond, the angle is positive when the far bond C-D is rotated
    clockwise from the near bond B-A. References in the opposite convention would
    restrain every torsion that is not symmetric under negation for its period
    (nucleotide and carbohydrate sugar rings among them) toward its mirror image.

    Parameters
    ----------
    xyz : torch.Tensor
        Cartesian coordinates in Å, shape (n_atoms, 3).
    idx : torch.Tensor
        Atom indices A, B, C, D of each dihedral, shape (n_torsions, 4), integer
        dtype.

    Returns
    -------
    torch.Tensor
        Dihedral angles in degrees in [-180, 180], shape (n_torsions,), in the dtype
        of ``xyz``. A fully degenerate quadruple (coincident or collinear atoms) gives
        0 with a zero gradient rather than NaN.
    """
    p1 = xyz[idx[:, 0]]
    p2 = xyz[idx[:, 1]]
    p3 = xyz[idx[:, 2]]
    p4 = xyz[idx[:, 3]]

    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    # Floor the |b2| divisor so collinear atoms (|b2| -> 0) give a finite
    # gradient instead of 0/0 = NaN.
    b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True).clamp_min(EPS)
    # b2_hat x n1, not n1 x b2_hat: the operand order is what makes the sign IUPAC.
    # Textbook forms built on n1 x b2_hat carry a compensating minus on the atan2.
    m1 = torch.cross(b2 / b2_norm, n1, dim=-1)

    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)
    # Guard atan2(0, 0) (fully degenerate dihedral, e.g. coincident atoms):
    # its backward is -y/(x^2+y^2), x/(x^2+y^2) = 0/0 = NaN. Replace such
    # entries with (x, y) = (1, 0) -> angle 0 with a finite (zero) gradient.
    degenerate = (x * x + y * y) < (EPS * EPS)
    x = torch.where(degenerate, torch.ones_like(x), x)
    y = torch.where(degenerate, torch.zeros_like(y), y)
    return torch.rad2deg(torch.atan2(y, x))
