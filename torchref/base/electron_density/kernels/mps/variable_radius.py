"""Metal (MPS) variable-radius electron-density splat, autograd-wrapped.

``add_isotropic_mps_var`` / ``add_anisotropic_mps_var`` mirror the portable CPU reference
signatures, so the dispatch site in ``main.py`` is a near-copy of the CPU branch. Each
delegates to a ``torch.autograd.Function`` dispatching the compiled Metal kernels (one
thread per atom) for forward and analytic backward.

Gradients flow to ``xyz``, ``adp``/``u`` and ``occ``, with identity to the input
``density_map``; ``A``/``B`` and the cell matrices get none, as on CUDA. **Backward is
first-order only** -- double backward must use ``force_portable``.
"""

from __future__ import annotations

import torch

from torchref.base.electron_density.kernels.mps.compile import _get_lib


def _r2cut(radius_per_atom):
    """Per-atom squared cutoff: the policy radius, used raw.

    Never round it up to a whole voxel. That makes the effective cutoff **grid-dependent**,
    inflating it by up to one voxel, so Metal truncates later than CUDA and
    ``torchref.sigma_cutoff_ed`` means different things on the two devices. The radius from
    :mod:`~torchref.base.electron_density.radius_policy` is already quantized and clamped.
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
    density_map, xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Isotropic variable-radius Metal splat; adds into ``density_map``.

    The canonical splat signature, identical to ``add_isotropic_plain_var`` and
    ``add_isotropic_cpu_sphere_var``: the grid shape comes from ``density_map`` and the
    truncation box from the inverse-cell row norms, so no ``grid_shape_tuple`` or
    ``voxel_size`` is taken. The radius is the policy radius used raw; see :func:`_r2cut`.
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
    density_map, xyz, u, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Anisotropic variable-radius Metal splat; adds into ``density_map``.

    The canonical splat signature, identical to ``add_anisotropic_plain_var`` and
    ``add_anisotropic_cpu_sphere_var``. Each atom is truncated at its per-axis bounding box
    with a sphere cull at the per-atom radius -- the same canonical cutoff the CUDA and
    fused-CPU kernels apply.
    """
    r2cut = _r2cut(radius_per_atom)
    mask = torch.ones(xyz.shape[0], 5, dtype=xyz.dtype, device=xyz.device)
    return MetalGridDensityAniso.apply(
        density_map, xyz, u, occ, A, B, r2cut, mask, inv_frac_matrix, frac_matrix
    )
