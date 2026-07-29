"""Metal (MPS) variable-radius electron-density splat, autograd-wrapped.

``add_isotropic_mps_var`` / ``add_anisotropic_mps_var`` mirror the signatures of
the portable CPU reference (``add_isotropic_plain_var`` /
``add_anisotropic_plain_var`` in ``kernels/cpu/variable_radius.py``) so the
dispatch site in ``main.py`` is a near-copy of the CPU branch. Each delegates to
a ``torch.autograd.Function`` that dispatches the compiled Metal kernels
(one thread per atom) for the forward and analytic backward.

Gradients flow to ``xyz``, ``adp``/``u``, and ``occ`` (and identity to the input
``density_map``); ``A``/``B`` and the cell matrices receive no gradient -- the
same set as the CUDA kernels. Backward is first-order only (like CUDA); double
backward must use ``Engine.EAGER`` (the plain splat).
"""

from __future__ import annotations

import torch

from torchref.base.electron_density.kernels.mps.compile import _get_lib


def _r2cut(radius_per_atom):
    """Per-atom squared cutoff: the policy radius, used raw.

    This used to round ``radius_per_atom`` up to a whole voxel, to match the old
    plain-splat reference whose offset list was built in voxel units. That made the
    effective cutoff **grid-dependent** -- inflating it by up to one voxel (~0.4 A,
    i.e. ~0.4 sigma at 0.4 A sampling), so Metal truncated systematically later than
    the CUDA kernel and ``torchref.sigma_cutoff_ed`` meant different things on the two
    devices. The radius arriving from
    :mod:`torchref.base.electron_density.radius_policy` is already quantized (0.25 A)
    and clamped, so it is used as-is; every backend now shares one cutoff.
    """
    return radius_per_atom * radius_per_atom


class MetalGridDensity(torch.autograd.Function):
    """Isotropic per-atom Metal splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, adp, occ, A, B, r2cut, mask, inv_frac, frac):
        lib = _get_lib()
        if lib is None:
            raise RuntimeError("Metal splat kernels unavailable")
        nx, ny, nz = (int(s) for s in density_map.shape)
        out = density_map.contiguous().clone()
        n = xyz.shape[0]
        if n > 0:
            lib.iso_splat_fwd(
                out.view(-1),
                xyz.contiguous(),
                adp.contiguous(),
                occ.contiguous(),
                A.contiguous(),
                B.contiguous(),
                r2cut.contiguous(),
                mask.contiguous(),
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                n, nx, ny, nz,
                threads=[n],
            )
        ctx.save_for_backward(xyz, adp, occ, A, B, r2cut, mask, inv_frac, frac)
        ctx.grid_shape = (nx, ny, nz)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xyz, adp, occ, A, B, r2cut, mask, inv_frac, frac = ctx.saved_tensors
        nx, ny, nz = ctx.grid_shape
        n = xyz.shape[0]
        grad_xyz = torch.zeros_like(xyz)
        grad_adp = torch.zeros_like(adp)
        grad_occ = torch.zeros_like(occ)
        if n > 0:
            lib = _get_lib()
            lib.iso_splat_bwd(
                grad_xyz.view(-1),
                grad_adp,
                grad_occ,
                grad_out.contiguous().view(-1),
                xyz.contiguous(),
                adp.contiguous(),
                occ.contiguous(),
                A.contiguous(),
                B.contiguous(),
                r2cut.contiguous(),
                mask.contiguous(),
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                n, nx, ny, nz,
                threads=[n],
            )
        # forward returned density_map + splat -> grad wrt density_map is identity.
        # order: density_map, xyz, adp, occ, A, B, r2cut, mask, inv_frac, frac
        return (grad_out, grad_xyz, grad_adp, grad_occ,
                None, None, None, None, None, None)


def add_isotropic_mps_var(
    density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, grid_shape_tuple, voxel_size, radius_per_atom,
):
    """Isotropic variable-radius Metal splat; adds into ``density_map``.

    Signature mirrors ``add_isotropic_plain_var`` (``grid_shape_tuple`` and
    ``voxel_size`` are unused -- the grid shape comes from ``density_map`` and the
    truncation box from the inverse-cell row norms).

    The truncation radius is the policy radius, used raw; see :func:`_r2cut`.
    """
    r2cut = _r2cut(radius_per_atom)
    mask = torch.ones(xyz.shape[0], 5, dtype=xyz.dtype, device=xyz.device)
    return MetalGridDensity.apply(
        density_map, xyz, adp, occ, A, B, r2cut, mask, inv_frac_matrix, frac_matrix
    )


class MetalGridDensityAniso(torch.autograd.Function):
    """Anisotropic per-atom Metal splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, u, occ, A, B, r2cut, mask, inv_frac, frac):
        lib = _get_lib()
        if lib is None:
            raise RuntimeError("Metal splat kernels unavailable")
        nx, ny, nz = (int(s) for s in density_map.shape)
        out = density_map.contiguous().clone()
        n = xyz.shape[0]
        if n > 0:
            lib.aniso_splat_fwd(
                out.view(-1),
                xyz.contiguous(),
                u.contiguous(),
                occ.contiguous(),
                A.contiguous(),
                B.contiguous(),
                r2cut.contiguous(),
                mask.contiguous(),
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                n, nx, ny, nz,
                threads=[n],
            )
        ctx.save_for_backward(xyz, u, occ, A, B, r2cut, mask, inv_frac, frac)
        ctx.grid_shape = (nx, ny, nz)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xyz, u, occ, A, B, r2cut, mask, inv_frac, frac = ctx.saved_tensors
        nx, ny, nz = ctx.grid_shape
        n = xyz.shape[0]
        grad_xyz = torch.zeros_like(xyz)
        grad_u = torch.zeros_like(u)
        grad_occ = torch.zeros_like(occ)
        if n > 0:
            lib = _get_lib()
            lib.aniso_splat_bwd(
                grad_xyz.view(-1),
                grad_u.view(-1),
                grad_occ,
                grad_out.contiguous().view(-1),
                xyz.contiguous(),
                u.contiguous(),
                occ.contiguous(),
                A.contiguous(),
                B.contiguous(),
                r2cut.contiguous(),
                mask.contiguous(),
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                n, nx, ny, nz,
                threads=[n],
            )
        # order: density_map, xyz, u, occ, A, B, r2cut, mask, inv_frac, frac
        return (grad_out, grad_xyz, grad_u, grad_occ,
                None, None, None, None, None, None)


def add_anisotropic_mps_var(
    real_space_grid, density_map, xyz, u, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_per_atom, voxel_size,
):
    """Anisotropic variable-radius Metal splat; adds into ``density_map``.

    Signature mirrors ``add_anisotropic_plain_var`` (``real_space_grid`` is used
    only for its grid shape). Each atom is truncated at its per-axis bounding box
    with a sphere cull at the per-atom radius -- the same canonical cutoff the
    CUDA and fused-CPU kernels apply.
    """
    r2cut = _r2cut(radius_per_atom)
    mask = torch.ones(xyz.shape[0], 5, dtype=xyz.dtype, device=xyz.device)
    return MetalGridDensityAniso.apply(
        density_map, xyz, u, occ, A, B, r2cut, mask, inv_frac_matrix, frac_matrix
    )
