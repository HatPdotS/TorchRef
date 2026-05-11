"""Shared helpers for target math kernels."""

import numpy as np
import torch

LOG_2PI: float = float(np.log(2.0 * np.pi))
DEG2RAD: float = float(np.pi) / 180.0
RAD2DEG: float = 180.0 / float(np.pi)


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
    m1 = torch.cross(
        n1, b2 / torch.linalg.norm(b2, dim=-1, keepdim=True), dim=-1
    )

    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)
    return torch.rad2deg(torch.atan2(y, x))
