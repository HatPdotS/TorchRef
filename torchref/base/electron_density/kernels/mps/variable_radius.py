"""Metal (MPS) variable-radius electron-density splat, autograd-wrapped.

``add_isotropic_mps_var`` / ``add_anisotropic_mps_var`` mirror the portable CPU reference
signatures, so the dispatch site in ``main.py`` is a near-copy of the CPU branch. Each
delegates to a ``torch.autograd.Function`` dispatching the compiled Metal kernels (one
thread per atom) for forward and analytic backward.

Gradients flow to ``xyz``, ``adp``/``u`` and ``occ``, with identity to the input
``density_map``; ``A``/``B`` and the cell matrices get none, as on CUDA. The backward is
differentiable once more: under ``create_graph=True`` it is recorded as
``_MetalSplatGrad``, whose backward runs the second-order shader (tangent map plus each
atom's Hessian-vector product). A third derivative raises.

A batched call (``(B, nx, ny, nz)`` map plus ``batch_offsets``, see
:mod:`~torchref.base.electron_density.batching`) hands the shaders each atom's member id.
Metal indexes in 32 bits, so a batch of ``2**31`` voxels or more is refused with a
``ValueError`` rather than wrapped.
"""

from __future__ import annotations

import torch

from torchref.base.electron_density.batching import check_batch, member_index
from torchref.base.electron_density.kernels.mps.compile import _get_lib
from torchref.base.targets._dispatch import first_order_only


def _r2cut(radius_per_atom):
    """Per-atom squared cutoff: the policy radius, used raw.

    Never round it up to a whole voxel. That makes the effective cutoff **grid-dependent**,
    inflating it by up to one voxel, so Metal truncates later than CUDA and
    ``torchref.sigma_cutoff_ed`` means different things on the two devices. The radius from
    :mod:`~torchref.base.electron_density.radius_policy` is already quantized and clamped.
    """
    return radius_per_atom * radius_per_atom


#: Largest voxel count, over the whole batch, that the shaders' ``int`` indices reach.
_MAX_VOXELS = 2**31 - 1


def _member(density_map, xyz, batch_offsets):
    """``(member_ids, has_batch)`` for the shaders.

    Unbatched calls pass a one-element placeholder the shader never reads, since a bound
    buffer is required either way.
    """
    _, offsets = check_batch(density_map, xyz.shape[0], batch_offsets)
    if density_map.numel() > _MAX_VOXELS:
        raise ValueError(
            f"the Metal splat indexes in 32 bits, and this map holds "
            f"{density_map.numel()} voxels (limit {_MAX_VOXELS}); split the batch"
        )
    if offsets is None:
        # dtype-ok: the shaders read member ids as int
        return torch.zeros(1, dtype=torch.int32, device=density_map.device), 0
    return member_index(offsets.to(density_map.device), xyz.shape[0]), 1


def _zeros_contiguous(t):
    # The shaders write per-atom arrays flat and row-major; zeros_like alone would keep a
    # strided input's layout and the gradient of a transposed xyz would come back permuted.
    return torch.zeros_like(t, memory_format=torch.contiguous_format)


class _MetalSplatGrad(torch.autograd.Function):
    """The splat's atom gradients as a function of the map gradient and the atoms.

    The backward of both Metal splats, ``kind`` naming the shader pair (``"iso"`` or
    ``"aniso"``). Autograd records it under ``create_graph=True``, and its own backward
    runs the ``*_splat_hvp`` shader; that backward is first-order only.
    """

    @staticmethod
    def forward(ctx, grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac, member,
                has_batch, kind, grid_shape):
        grads = tuple(_zeros_contiguous(t) for t in (xyz, third, occ))
        n = xyz.shape[0]
        if n > 0:
            getattr(_get_lib(), f"{kind}_splat_bwd")(
                grads[0].view(-1),
                grads[1].view(-1),
                grads[2],
                grad_out.contiguous().view(-1),
                xyz.contiguous(),
                third.contiguous(),
                occ.contiguous(),
                A.contiguous(),
                B.contiguous(),
                r2cut.contiguous(),
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                n, *grid_shape, member, has_batch,
                threads=[n],
            )
        ctx.has_batch, ctx.kind, ctx.grid_shape = has_batch, kind, grid_shape
        ctx.save_for_backward(grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac,
                              member)
        return grads

    @staticmethod
    @first_order_only
    def backward(ctx, t_xyz, t_third, t_occ):
        grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac, member = ctx.saved_tensors
        atoms = (xyz, third, occ)
        # A cotangent autograd leaves out (None) is zero.
        cotangents = tuple(
            _zeros_contiguous(x) if t is None else t.contiguous()
            for t, x in zip((t_xyz, t_third, t_occ), atoms)
        )
        hvps = tuple(_zeros_contiguous(x) for x in atoms)
        # The tangent map is the gradient wrt grad_out, wanted only when the map gradient
        # itself depends on the atoms (any loss nonlinear in the density).
        write_tangent = ctx.needs_input_grad[0]
        tangent = _zeros_contiguous(grad_out) if write_tangent else None
        n = xyz.shape[0]
        if n > 0:
            getattr(_get_lib(), f"{ctx.kind}_splat_hvp")(
                # A bound buffer is required even when the shader skips the writes.
                grad_out.new_zeros(1) if tangent is None else tangent.view(-1),
                hvps[0].view(-1),
                hvps[1].view(-1),
                hvps[2],
                grad_out.contiguous().view(-1),
                xyz.contiguous(),
                third.contiguous(),
                occ.contiguous(),
                A.contiguous(),
                B.contiguous(),
                r2cut.contiguous(),
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                cotangents[0].view(-1),
                cotangents[1].view(-1),
                cotangents[2],
                n, *ctx.grid_shape, member, ctx.has_batch, int(write_tangent),
                threads=[n],
            )
        # order: grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac, member, has_batch,
        # kind, grid_shape
        return (tangent, *hvps) + (None,) * 9


def _backward(ctx, kind, grad_out):
    """Backward shared by both splats; the gradient of ``third`` has its shape."""
    xyz, third, occ, A, B, r2cut, inv_frac, frac, member = ctx.saved_tensors
    grads = _MetalSplatGrad.apply(grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac,
                                  member, ctx.has_batch, kind, ctx.grid_shape)
    # forward returned density_map + splat -> grad wrt density_map is identity.
    # order: density_map, xyz, third, occ, A, B, r2cut, inv_frac, frac, member, has_batch
    return (grad_out, *grads, None, None, None, None, None, None, None)


class MetalGridDensity(torch.autograd.Function):
    """Isotropic per-atom Metal splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, adp, occ, A, B, r2cut, inv_frac, frac,
                member, has_batch):
        lib = _get_lib()
        if lib is None:
            raise RuntimeError("Metal splat kernels unavailable")
        nx, ny, nz = (int(s) for s in density_map.shape[-3:])
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
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                n, nx, ny, nz, member, has_batch,
                threads=[n],
            )
        ctx.save_for_backward(xyz, adp, occ, A, B, r2cut, inv_frac, frac, member)
        ctx.grid_shape = (nx, ny, nz)
        ctx.has_batch = has_batch
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return _backward(ctx, "iso", grad_out)


def add_isotropic_mps_var(
    density_map, xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom,
    batch_offsets=None,
):
    """Isotropic variable-radius Metal splat; returns ``density_map + splat``.

    ``density_map`` is not modified; it is ``(nx, ny, nz)``, or ``(B, nx, ny, nz)`` with
    ``batch_offsets`` ``(B + 1,)``.

    The canonical splat signature, identical to ``add_isotropic_plain_var`` and
    ``add_isotropic_cpu_sphere_var``: the grid shape comes from ``density_map`` and the
    truncation box from the inverse-cell row norms, so no ``grid_shape_tuple`` or
    ``voxel_size`` is taken. The radius is the policy radius used raw; see :func:`_r2cut`.
    """
    r2cut = _r2cut(radius_per_atom)
    member, has_batch = _member(density_map, xyz, batch_offsets)
    return MetalGridDensity.apply(
        density_map, xyz, adp, occ, A, B, r2cut, inv_frac_matrix, frac_matrix,
        member, has_batch,
    )


class MetalGridDensityAniso(torch.autograd.Function):
    """Anisotropic per-atom Metal splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, u, occ, A, B, r2cut, inv_frac, frac,
                member, has_batch):
        lib = _get_lib()
        if lib is None:
            raise RuntimeError("Metal splat kernels unavailable")
        nx, ny, nz = (int(s) for s in density_map.shape[-3:])
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
                inv_frac.contiguous().view(-1),
                frac.contiguous().view(-1),
                n, nx, ny, nz, member, has_batch,
                threads=[n],
            )
        ctx.save_for_backward(xyz, u, occ, A, B, r2cut, inv_frac, frac, member)
        ctx.grid_shape = (nx, ny, nz)
        ctx.has_batch = has_batch
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return _backward(ctx, "aniso", grad_out)


def add_anisotropic_mps_var(
    density_map, xyz, u, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom,
    batch_offsets=None,
):
    """Anisotropic variable-radius Metal splat; returns ``density_map + splat``.

    ``density_map`` is not modified.

    The canonical splat signature, identical to ``add_anisotropic_plain_var`` and
    ``add_anisotropic_cpu_sphere_var``. Each atom is truncated at its per-axis bounding box
    with a sphere cull at the per-atom radius -- the same canonical cutoff the CUDA and
    fused-CPU kernels apply.
    """
    r2cut = _r2cut(radius_per_atom)
    member, has_batch = _member(density_map, xyz, batch_offsets)
    return MetalGridDensityAniso.apply(
        density_map, xyz, u, occ, A, B, r2cut, inv_frac_matrix, frac_matrix,
        member, has_batch,
    )
