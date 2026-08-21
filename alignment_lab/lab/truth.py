"""Seeded rotations and rank-of-truth against a symmetry orbit.

Two things here previously existed in several disagreeing copies, and both
silently changed results rather than raising:

* **The rotation generator.** Two QR-based variants were in circulation; the
  one omitting the ``sign(diag(R))`` correction is not Haar-uniform and returns
  a *different* rotation for the same seed. Runs from the two families are not
  comparable. :func:`random_rotation` is the corrected form.
* **The orbit convention.** Rank-of-truth was computed with the symmetry
  operators applied on either side, and in either the fractional or the
  Cartesian frame. The choice changes the answer, so it is an explicit argument
  here and is meant to be recorded in every result row.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch


def random_rotation(seed: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Haar-uniform random rotation matrix from a seed.

    QR of a Gaussian matrix, with the ``sign(diag(R))`` correction that makes
    the decomposition unique -- without it the distribution is not Haar-uniform
    and the seed maps to a different rotation.

    Parameters
    ----------
    seed : int
        Generator seed. The mapping seed -> rotation is the reproducibility
        contract for the whole lab; changing this function invalidates every
        archived result.
    dtype : torch.dtype, optional
        Output dtype. Default ``torch.float64``.

    Returns
    -------
    torch.Tensor
        ``(3, 3)`` rotation with ``det = +1``.
    """
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def seed_for(pdb: str, trial: int, base: int = 42) -> int:
    """Seed for a ``(structure, trial)`` cell of the benchmark.

    ``base + 1000 * trial + index(pdb) * 7`` -- the convention the archived
    results were produced under. The index term is why
    :data:`~alignment_lab.lab.benchmark.BENCH_PDBS` is append-only.

    Parameters
    ----------
    pdb : str
        Benchmark structure code.
    trial : int
        Trial number.
    base : int, optional
        Seed base. Default 42.

    Returns
    -------
    int
        The seed.
    """
    from .benchmark import BENCH_PDBS

    return int(base) + 1000 * int(trial) + BENCH_PDBS.index(pdb) * 7


def symmetry_orbit(
    R_true: torch.Tensor,
    symops: torch.Tensor,
    *,
    side: str = "left",
    frame: str = "cart",
    reciprocal_basis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the set of rotations equivalent to ``R_true`` under the point group.

    Parameters
    ----------
    R_true : torch.Tensor
        ``(3, 3)`` true rotation.
    symops : torch.Tensor
        ``(n_ops, 3, 3)`` symmetry rotation parts, as stored on the space group
        (fractional).
    side : {'left', 'right'}, optional
        ``'left'`` builds ``S_k @ R_true``; ``'right'`` builds ``R_true @ S_k``.
        These are different sets for non-commuting operators.
    frame : {'cart', 'frac'}, optional
        ``'cart'`` converts the operators to the Cartesian frame first, which is
        the frame the rotation function works in. ``'frac'`` uses them as
        stored. Mixing a Cartesian rotation with fractional operators is a
        metric error that inflates apparent ghost counts.
    reciprocal_basis : torch.Tensor, optional
        ``(3, 3)`` reciprocal basis, required when ``frame='cart'``.

    Returns
    -------
    torch.Tensor
        ``(n_ops, 3, 3)`` orbit members, float64.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if frame not in ("cart", "frac"):
        raise ValueError(f"frame must be 'cart' or 'frac', got {frame!r}")

    R = R_true.to(torch.float64)
    S = symops.to(torch.float64)
    if frame == "cart":
        if reciprocal_basis is None:
            raise ValueError("frame='cart' requires reciprocal_basis")
        from torchref.experimental.alignment.sh import hkl_symops_to_cartesian

        S = hkl_symops_to_cartesian(S, reciprocal_basis.to(torch.float64))
    return S @ R.unsqueeze(0) if side == "left" else R.unsqueeze(0) @ S


def angle_to_orbit(R: torch.Tensor, orbit: torch.Tensor) -> float:
    """Smallest rotation angle between ``R`` and any orbit member, in degrees.

    Parameters
    ----------
    R : torch.Tensor
        ``(3, 3)`` rotation.
    orbit : torch.Tensor
        ``(n, 3, 3)`` orbit members.

    Returns
    -------
    float
        Angle in degrees.
    """
    tr = torch.einsum("kij,ij->k", orbit.to(torch.float64), R.to(torch.float64))
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(cos.arccos().min() * (180.0 / math.pi))


def orbit_rank(
    peaks: Sequence,
    R_true: torch.Tensor,
    symops: torch.Tensor,
    *,
    side: str = "left",
    frame: str = "cart",
    reciprocal_basis: Optional[torch.Tensor] = None,
    thr_deg: float = 5.0,
) -> Tuple[int, float]:
    """Rank of the first peak matching the true orientation.

    Parameters
    ----------
    peaks : sequence
        Peaks carrying Edmonds ZYZ ``alpha``/``beta``/``gamma`` in radians, in
        descending score order (the FRF's ``RotationPeak``).
    R_true : torch.Tensor
        ``(3, 3)`` true rotation.
    symops : torch.Tensor
        ``(n_ops, 3, 3)`` symmetry rotation parts.
    side, frame, reciprocal_basis
        Orbit convention -- see :func:`symmetry_orbit`. Record these alongside
        any rank you report; the rank is meaningless without them.
    thr_deg : float, optional
        Match threshold in degrees. Default 5.0.

    Returns
    -------
    tuple
        ``(rank, best_angle_deg)``. ``rank`` is ``-1`` when no peak matches;
        ``best_angle_deg`` is the closest approach over all peaks either way,
        which distinguishes "just outside the threshold" from "absent".
    """
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_matrix_from_edmonds_euler,
    )

    orbit = symmetry_orbit(
        R_true, symops, side=side, frame=frame, reciprocal_basis=reciprocal_basis,
    )
    rank, best = -1, float("inf")
    for i, p in enumerate(peaks):
        R_p = rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma)
        ang = angle_to_orbit(R_p, orbit)
        if ang < best:
            best = ang
        if ang <= thr_deg and rank < 0:
            rank = i
    return rank, best
