"""Non-bonded (VDW) repulsion NLL — PROLSQ form with symmetry mates.

:func:`nonbonded_pair_positions` places both ends of every pair. The eager kernel
here and the statistics of ``NonBondedTarget`` and ``NonBondedHTarget`` all read their
positions from it; the Triton kernel (:mod:`torchref.base.targets.triton.nonbonded`)
computes the same positions in-kernel.
"""

from typing import Optional

import torch

from torchref.base.coordinates.symmetry_images import symmetry_image_positions

from ._common import LOG_2PI
from ._dispatch import use_triton


def nonbonded_pair_positions(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    symop_indices: torch.Tensor | None,
    cell_offsets: torch.Tensor | None,
    symop_matrices: torch.Tensor | None,
    symop_translations: torch.Tensor | None,
    fractional_matrix: torch.Tensor | None,
    inv_fractional_matrix: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cartesian positions of both ends of every non-bonded pair.

    The first atom of a pair is the asymmetric-unit atom itself; the second is the
    image ``(symop, cell_offset)`` of an ASU atom, placed by
    :func:`~torchref.base.coordinates.symmetry_images.symmetry_image_positions` from the
    current coordinates, so gradients reach both atoms.

    Parameters
    ----------
    xyz : torch.Tensor
        Cartesian ASU coordinates in Å, shape ``(N_atoms, 3)``.
    indices : torch.Tensor
        Atom indices per pair, ``(N, 2)`` integer.
    symop_indices : torch.Tensor or None
        Operation index per pair, ``(N,)`` integer; 0 is the identity. None means
        every partner is the ASU atom itself, and the four symmetry arguments below
        are then not read.
    cell_offsets : torch.Tensor or None
        Integer lattice translation per pair, ``(N, 3)``, in fractional units,
        relative to ``xyz`` as given (unwrapped). None means no translation.
    symop_matrices, symop_translations : torch.Tensor or None
        The operation table: ``(n_ops, 3, 3)`` rotations and ``(n_ops, 3)``
        translations, in the fractional basis.
    fractional_matrix, inv_fractional_matrix : torch.Tensor or None
        ``Cell.fractional_matrix`` and its inverse, ``(3, 3)``.

    Returns
    -------
    pos1, pos2 : torch.Tensor
        Each ``(N, 3)`` in Å, Cartesian.

    Notes
    -----
    With ``symop_indices`` given, every pair is transformed, intra-ASU ones included
    (they come out as the identity). There is no shortcut for lists whose operations
    are all 0: those still carry lattice translations, and in P1 they are all of the
    crystal contacts.
    """
    pos1 = xyz[indices[:, 0]]
    partner = xyz[indices[:, 1]]
    if symop_indices is None:
        return pos1, partner
    pos2 = symmetry_image_positions(
        partner,
        symop_indices,
        cell_offsets,
        symop_matrices,
        symop_translations,
        fractional_matrix,
        inv_fractional_matrix,
    )
    return pos1, pos2


def _nonbonded_heavy_math_eager(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    min_distances: torch.Tensor,
    symop_indices: Optional[torch.Tensor],
    cell_offsets: Optional[torch.Tensor],
    symop_matrices: Optional[torch.Tensor],
    symop_translations: Optional[torch.Tensor],
    fractional_matrix: torch.Tensor,
    inv_fractional_matrix: torch.Tensor,
    c_rep: torch.Tensor,
    r_exp: torch.Tensor,
    buffer: float,
    sigma_vdw: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pos1, pos2 = nonbonded_pair_positions(
        xyz,
        indices,
        symop_indices,
        cell_offsets,
        symop_matrices,
        symop_translations,
        fractional_matrix,
        inv_fractional_matrix,
    )
    diff = pos2 - pos1
    actual_distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    violations = torch.clamp(min_distances + buffer - actual_distances, min=0.0)
    shape_energy = c_rep * (violations ** r_exp)
    per_pair_const = torch.log(sigma_vdw) + 0.5 * LOG_2PI
    if weights is None:
        return shape_energy.sum() + per_pair_const * violations.shape[0]
    return (weights * shape_energy).sum() + per_pair_const * weights.sum()


def nonbonded_heavy_math(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    min_distances: torch.Tensor,
    symop_indices: Optional[torch.Tensor],
    cell_offsets: Optional[torch.Tensor],
    symop_matrices: Optional[torch.Tensor],
    symop_translations: Optional[torch.Tensor],
    fractional_matrix: torch.Tensor,
    inv_fractional_matrix: torch.Tensor,
    c_rep: torch.Tensor,
    r_exp: torch.Tensor,
    buffer: float,
    sigma_vdw: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Heavy-heavy VDW prolsq repulsion NLL.

    The loss of ``NonBondedTarget.forward``, with pair positions from
    :func:`nonbonded_pair_positions`. The H-VDW term ``NonBondedHTarget``
    adds is **not** included here. Dispatches to
    :func:`torchref.base.targets.triton.nonbonded.nonbonded_heavy_math_triton` on CUDA
    float32 (the gain coming mostly from the analytic backward), eager otherwise.

    Parameters
    ----------
    xyz : torch.Tensor
        (N_atoms, 3) Cartesian coordinates of the ASU in Å.
    indices : torch.Tensor
        (N, 2) per-pair atom indices.
    min_distances : torch.Tensor
        (N,) VDW threshold per pair in Å.
    symop_indices : torch.Tensor, optional
        (N,) symmetry-operator index per pair; 0 = identity. None treats every
        partner as the ASU atom itself; otherwise every pair is imaged, see
        :func:`nonbonded_pair_positions`.
    cell_offsets : torch.Tensor, optional
        (N, 3) integer lattice translations per pair, in fractional units.
    symop_matrices, symop_translations : torch.Tensor, optional
        (n_symops, 3, 3) and (n_symops, 3) — the symmetry operator table.
    fractional_matrix, inv_fractional_matrix : torch.Tensor
        ``cell.fractional_matrix`` and its inverse (3, 3).
    c_rep, r_exp, sigma_vdw : torch.Tensor
        Scalar repulsion coefficient, exponent, and effective tolerance.
    buffer : float
        Distance buffer in Å.
    weights : torch.Tensor, optional
        (N,) weight of each pair's NLL, its constant included. None weighs every
        pair 1; the pair lists carry the weights to pass (``'weights'`` of the VDW
        restraints, ``HydrogenTopology.cand_weight``), which count a crystal contact,
        listed from both of its ends, once.
    """
    if use_triton(xyz):
        from .triton.nonbonded import nonbonded_heavy_math_triton

        return nonbonded_heavy_math_triton(
            xyz,
            indices,
            min_distances,
            symop_indices,
            cell_offsets,
            symop_matrices,
            symop_translations,
            fractional_matrix,
            inv_fractional_matrix,
            c_rep,
            r_exp,
            buffer,
            sigma_vdw,
            weights=weights,
        )
    return _nonbonded_heavy_math_eager(
        xyz,
        indices,
        min_distances,
        symop_indices,
        cell_offsets,
        symop_matrices,
        symop_translations,
        fractional_matrix,
        inv_fractional_matrix,
        c_rep,
        r_exp,
        buffer,
        sigma_vdw,
        weights,
    )
