"""Positions of atoms under a symmetry operation followed by a lattice translation.

An image is named by a pair ``(symop, cell_offset)``: operation ``symop`` of the space
group, ``x -> R x + t`` in fractional coordinates, then the integer lattice translation
``cell_offset``. :func:`symmetry_image_positions` is the one place such a pair becomes a
Cartesian position, and :func:`is_symmetry_image` the one test for "this is an image,
not the atom itself". The non-bonded pair builder (:mod:`torchref.topology.nonbonded`)
forms the images it searches with the first, and the non-bonded scoring
(:func:`torchref.base.targets.nonbonded.nonbonded_pair_positions`) places the stored
pairs with it, so a pair is scored at the distance it was found at.

Offsets apply to the coordinates as given; nothing here wraps into the unit cell. A
stored ``cell_offset`` therefore stays valid only together with the unwrapped
coordinates it was chosen for.
"""

import torch


def symmetry_image_positions(
    xyz: torch.Tensor,
    symop_indices: torch.Tensor,
    cell_offsets: torch.Tensor | None,
    symop_matrices: torch.Tensor,
    symop_translations: torch.Tensor,
    fractional_matrix: torch.Tensor,
    inv_fractional_matrix: torch.Tensor,
) -> torch.Tensor:
    """Place each point's image: ``B (R_s B^-1 x + t_s + n)``.

    Parameters
    ----------
    xyz : torch.Tensor
        Cartesian source positions in Å, shape ``(..., 3)``.
    symop_indices : torch.Tensor
        Integer index ``s`` into the operation table per point, any shape that
        broadcasts against ``xyz[..., 0]``; 0 is the identity.
    cell_offsets : torch.Tensor or None
        Integer lattice translation ``n`` per point, in fractional units, any shape
        that broadcasts against ``xyz``. None means no translation.
    symop_matrices : torch.Tensor
        Rotation part of every operation in the fractional basis, ``(n_ops, 3, 3)``.
    symop_translations : torch.Tensor
        Fractional translation part of every operation, ``(n_ops, 3)``.
    fractional_matrix : torch.Tensor
        Orthogonalization matrix ``B`` (fractional to Cartesian), ``(3, 3)``, as
        :attr:`torchref.symmetry.cell.Cell.fractional_matrix`.
    inv_fractional_matrix : torch.Tensor
        Its inverse, ``(3, 3)``, as :attr:`~.Cell.inv_fractional_matrix`.

    Returns
    -------
    torch.Tensor
        Cartesian image positions in Å, of the broadcast shape ``(..., 3)`` and in
        ``xyz``'s dtype. Differentiable in ``xyz`` and in both cell matrices.

    Notes
    -----
    Every point is transformed, identity entries included: ``(0, 0)`` returns ``xyz``
    up to the rounding of the ``B^-1``/``B`` round trip. There is deliberately no
    shortcut for lists whose operations are all the identity, because such a list can
    still carry lattice translations -- in P1 every crystal contact is one.
    """
    dtype = xyz.dtype
    frac = xyz @ inv_fractional_matrix.to(dtype).T
    rot = symop_matrices.to(dtype)[symop_indices]
    # einsum rather than a broadcast matmul: with xyz (N, 1, 3) against M operations,
    # matmul copies one 3x3 matrix per output point, einsum only writes the output.
    frac_image = torch.einsum("...ij,...j->...i", rot, frac)
    frac_image = frac_image + symop_translations.to(dtype)[symop_indices]
    if cell_offsets is not None:
        frac_image = frac_image + cell_offsets.to(dtype)
    return frac_image @ fractional_matrix.to(dtype).T


def is_symmetry_image(
    symop_indices: torch.Tensor, cell_offsets: torch.Tensor
) -> torch.Tensor:
    """Whether each ``(symop, cell_offset)`` names an image rather than the atom itself.

    True wherever the operation is not the identity *or* the lattice translation is
    nonzero: a pure lattice translation is a real image, and the only kind P1 has.

    Parameters
    ----------
    symop_indices : torch.Tensor
        Integer operation index per entry, shape ``(...)``; 0 is the identity.
    cell_offsets : torch.Tensor
        Integer lattice translation per entry, shape ``(..., 3)``.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``symop_indices.shape``.
    """
    return (symop_indices != 0) | (cell_offsets != 0).any(dim=-1)
