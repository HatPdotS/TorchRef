"""Shared helpers for target math kernels."""

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

    Matches the sign convention of ``Restraints.torsions``.

    Parameters
    ----------
    xyz : torch.Tensor
        (N_atoms, 3) Cartesian coordinates.
    idx : torch.Tensor
        (N, 4) atom indices defining each dihedral.
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
    m1 = torch.cross(n1, b2 / b2_norm, dim=-1)

    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)
    # Guard atan2(0, 0) (fully degenerate dihedral, e.g. coincident atoms):
    # its backward is -y/(x^2+y^2), x/(x^2+y^2) = 0/0 = NaN. Replace such
    # entries with (x, y) = (1, 0) -> angle 0 with a finite (zero) gradient.
    degenerate = (x * x + y * y) < (EPS * EPS)
    x = torch.where(degenerate, torch.ones_like(x), x)
    y = torch.where(degenerate, torch.zeros_like(y), y)
    return torch.rad2deg(torch.atan2(y, x))
