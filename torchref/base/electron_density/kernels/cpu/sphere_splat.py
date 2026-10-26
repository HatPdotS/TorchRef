"""Fused per-atom spherical-cutoff density splat for CPU, with autograd.

The production CPU path for :func:`build_electron_density`. The kernels are prebuilt Rust
(``torchref-kernels``, see :mod:`torchref.utils.native`) and a deliberate transliteration
of the Metal kernels in ``kernels/mps/_shaders.py``: CPU, CUDA and Metal implement *one*
truncation contract, so ``torchref.sigma_cutoff_ed`` means the same thing on every device.
Canonically:

    voxel v receives atom i's full 5-Gaussian density iff  ||w||^2 <= r_i^2,

where ``w`` is the minimum-image **Cartesian atom->voxel** vector (sphere centred on the
atom, not on its nearest grid node) and ``r_i`` is the raw ``radius_policy`` radius,
enumerated over the triclinic-correct per-axis half-width
``ceil(r_i * n_axis * ||inv_frac row_axis||)``. No grid-dependent requantization of the
radius, no diagonal metric.

Forward partitions the **output** by x-plane and backward partitions over **atoms**, so
neither needs atomics and both are independent of the thread count, which follows
``torch.get_num_threads()``. float32 uses a branchless ``fast_exp`` (the CPU analogue of
``metal::fast::exp``), whose disagreement with ``exp`` is far below the
amplitude-truncation floor at the default cutoff; float64 uses libm ``exp``, a float64
caller being precision-motivated by definition.

Gradients flow to ``xyz``, ``adp``/``u`` and ``occ`` with identity to the incoming
``density_map``; ``A``/``B`` and the cell matrices get none, as in the CUDA and Metal
kernels. **Backward is first-order only**; a ``create_graph=True`` backward is re-derived
through the portable splat (:func:`_double_backward_vjp`).

A batched call (``(B, nx, ny, nz)`` map plus ``batch_offsets``, see
:mod:`~torchref.base.electron_density.batching`) runs each member through the same
kernels, so it is bitwise the stack of the members' unbatched results.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from torchref.base.electron_density.batching import check_batch, dense_offsets
from torchref.utils import native


_INPUTS = ("xyz", "adp", "occ", "A", "B", "r2cut", "inv_frac", "frac")
_ARG_NAMES = {
    "fwd": ("out",) + _INPUTS,
    "bwd": ("g_xyz", "g_adp", "g_occ", "grad") + _INPUTS,
}
# dtype-ok: the kernels read member offsets as i64
_OFFSETS_DTYPE = torch.int64


class _RustSplat:
    """Tensor-level adapter over the ``torchref-kernels`` splat entry points.

    Each method takes the entry point's buffers as tensors, then the int64 member
    ``offsets``, then ``nx, ny, nz``. Every other tensor must already be contiguous and of
    the first one's dtype; :func:`torchref.utils.native.buf` checks that before any
    pointer is handed over.
    """

    def __init__(self, mod):
        self._mod = mod

    def _call(self, name, *args):
        *tensors, offsets, nx, ny, nz = args
        dtype = tensors[0].dtype
        suffix = {torch.float32: "f32", torch.float64: "f64"}.get(dtype)  # dtype-ok: kernel dtype dispatch, not an allocation
        if suffix is None:
            raise ValueError(f"sphere_splat supports float32/float64, got {dtype}")
        names = _ARG_NAMES[name.split("_")[1]]
        bufs = [native.buf(t, dtype, n) for n, t in zip(names, tensors, strict=True)]
        bufs.append(native.buf(offsets, _OFFSETS_DTYPE, "offsets"))
        fn = getattr(self._mod, f"sphere_{name}_{suffix}")
        fn(*bufs, nx, ny, nz, native.num_threads())

    iso_fwd = functools.partialmethod(_call, "iso_fwd")
    iso_bwd = functools.partialmethod(_call, "iso_bwd")
    aniso_fwd = functools.partialmethod(_call, "aniso_fwd")
    aniso_bwd = functools.partialmethod(_call, "aniso_bwd")


_module: Optional[_RustSplat] = None


def _get_module() -> Optional[_RustSplat]:
    """The splat kernels, or None if ``torchref-kernels`` is unavailable."""
    global _module
    if _module is None:
        mod = native.native()
        if mod is not None:
            _module = _RustSplat(mod)
    return _module


def why_unavailable() -> Optional[str]:
    """``None`` if the fused CPU splat is usable, else why it is not.

    The single availability probe for this backend, consumed by
    :mod:`torchref.utils.backends`.
    """
    if _get_module() is not None:
        return None
    reason = native.why_unavailable() or "unknown reason"
    return f"the fused CPU sphere_splat kernel is not available: {reason}"


def last_error() -> Optional[Tuple[str, str]]:
    """The ``(message, traceback)`` of the kernel load failure, if any."""
    return native.last_error()


def _require_module() -> _RustSplat:
    mod = _get_module()
    if mod is None:
        raise RuntimeError(why_unavailable())
    return mod


def _double_backward_vjp(plain_fn, ctx, grad_out, leaves, statics, r2cut, offsets):
    """Recompute this VJP through the portable splat for a ``create_graph=True`` backward.

    The native backward has no autograd graph, so the backward methods call this instead when
    a second derivative may be taken. Gradients are taken with respect to the saved (not
    detached) leaves, so the result stays on the caller's graph; the gradient with respect
    to ``density_map`` is the identity.
    """
    A, B, inv_frac, frac = statics
    # Only the leaves that actually require grad may be differentiated; asking for
    # the others raises regardless of allow_unused.
    wanted = [t for t in leaves if t.requires_grad]
    out = [None] * len(leaves)
    # With no atoms the portable splat returns the zero map with no graph to
    # differentiate; None is autograd's zero gradient.
    if wanted and leaves[0].shape[0] > 0:
        zeros = torch.zeros(
            ctx.map_shape, dtype=grad_out.dtype, device=grad_out.device
        )
        with torch.enable_grad():
            dm = plain_fn(
                zeros, *leaves, A, B, inv_frac, frac, r2cut.sqrt(),
                batch_offsets=offsets if ctx.batched else None,
            )
        grads = torch.autograd.grad(
            dm, wanted, grad_out, create_graph=True, allow_unused=True
        )
        it = iter(grads)
        out = [next(it) if t.requires_grad else None for t in leaves]
    # forward returned density_map + splat -> grad wrt density_map is the identity
    return (grad_out, *out) + (None,) * 6


def _prep(density_map, xyz, radius_per_atom, batch_offsets, *tensors):
    """Shared validation + contiguity for both entry points.

    Returns the squared radii and the member offsets, ``[0, n]`` for an unbatched call.
    """
    if density_map.device.type != "cpu":
        raise ValueError(
            f"sphere_splat is a CPU kernel; got device {density_map.device}"
        )
    dtype = density_map.dtype
    if dtype not in (torch.float32, torch.float64):  # dtype-ok: validation guard, not an allocation
        raise ValueError(f"sphere_splat supports float32/float64, got {dtype}")
    for t in (xyz, radius_per_atom) + tensors:
        if t.dtype != dtype:
            raise ValueError(
                f"every input must match density_map.dtype ({dtype}); got {t.dtype}"
            )
    r2cut = (radius_per_atom * radius_per_atom).detach().contiguous()
    _, offsets = check_batch(density_map, xyz.shape[0], batch_offsets)
    if offsets is None:
        offsets = dense_offsets(1, xyz.shape[0], density_map.device)
    return r2cut, offsets.contiguous()


class _FusedIsoSplat(torch.autograd.Function):
    """Isotropic fused CPU splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, adp, occ, A, B, r2cut, inv_frac, frac,
                offsets):
        mod = _require_module()
        nx, ny, nz = (int(s) for s in density_map.shape[-3:])
        out = density_map.contiguous().clone()
        if xyz.shape[0] > 0:
            mod.iso_fwd(
                out.view(-1),
                xyz.detach().contiguous(), adp.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                offsets, nx, ny, nz,
            )
        ctx.save_for_backward(xyz, adp, occ, A, B, r2cut, inv_frac, frac, offsets)
        ctx.grid_shape = (nx, ny, nz)
        ctx.map_shape = tuple(density_map.shape)
        ctx.batched = density_map.ndim == 4
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xyz, adp, occ, A, B, r2cut, inv_frac, frac, offsets = ctx.saved_tensors
        nx, ny, nz = ctx.grid_shape
        # Autograd runs backward under no_grad unless create_graph=True; grad_out's
        # requires_grad cannot tell, since a top-level create_graph seed is a plain tensor.
        if torch.is_grad_enabled():
            from torchref.base.electron_density.kernels.cpu.variable_radius import (
                add_isotropic_plain_var,
            )

            return _double_backward_vjp(
                add_isotropic_plain_var, ctx, grad_out,
                (xyz, adp, occ), (A, B, inv_frac, frac), r2cut, offsets,
            )
        g_xyz = torch.zeros_like(xyz)
        g_adp = torch.zeros_like(adp)
        g_occ = torch.zeros_like(occ)
        if xyz.shape[0] > 0:
            _require_module().iso_bwd(
                g_xyz.view(-1), g_adp, g_occ,
                grad_out.contiguous().view(-1),
                xyz.detach().contiguous(), adp.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                offsets, nx, ny, nz,
            )
        # out = density_map + splat, so grad wrt density_map is the identity.
        return (grad_out, g_xyz, g_adp, g_occ, None, None, None, None, None, None)


class _FusedAnisoSplat(torch.autograd.Function):
    """Anisotropic fused CPU splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, u, occ, A, B, r2cut, inv_frac, frac,
                offsets):
        mod = _require_module()
        nx, ny, nz = (int(s) for s in density_map.shape[-3:])
        out = density_map.contiguous().clone()
        if xyz.shape[0] > 0:
            mod.aniso_fwd(
                out.view(-1),
                xyz.detach().contiguous(), u.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                offsets, nx, ny, nz,
            )
        ctx.save_for_backward(xyz, u, occ, A, B, r2cut, inv_frac, frac, offsets)
        ctx.grid_shape = (nx, ny, nz)
        ctx.map_shape = tuple(density_map.shape)
        ctx.batched = density_map.ndim == 4
        return out

    @staticmethod
    def backward(ctx, grad_out):
        xyz, u, occ, A, B, r2cut, inv_frac, frac, offsets = ctx.saved_tensors
        nx, ny, nz = ctx.grid_shape
        if torch.is_grad_enabled():  # see _FusedIsoSplat.backward
            from torchref.base.electron_density.kernels.cpu.variable_radius import (
                add_anisotropic_plain_var,
            )

            return _double_backward_vjp(
                add_anisotropic_plain_var, ctx, grad_out,
                (xyz, u, occ), (A, B, inv_frac, frac), r2cut, offsets,
            )
        g_xyz = torch.zeros_like(xyz)
        g_u = torch.zeros_like(u)
        g_occ = torch.zeros_like(occ)
        if xyz.shape[0] > 0:
            _require_module().aniso_bwd(
                g_xyz.view(-1), g_u.view(-1), g_occ,
                grad_out.contiguous().view(-1),
                xyz.detach().contiguous(), u.detach().contiguous(),
                occ.detach().contiguous(), A.contiguous(), B.contiguous(),
                r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
                offsets, nx, ny, nz,
            )
        return (grad_out, g_xyz, g_u, g_occ, None, None, None, None, None, None)


def add_isotropic_cpu_sphere_var(
    density_map, xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom,
    batch_offsets=None,
):
    """Fused isotropic spherical-cutoff splat; returns ``density_map + splat``.

    Parameters
    ----------
    density_map : torch.Tensor
        Running map, shape ``(nx, ny, nz)``, or ``(B, nx, ny, nz)`` for a batch of
        members, CPU float32/float64. **Not mutated.**
    xyz, adp, occ : torch.Tensor
        Cartesian positions ``(n, 3)``, isotropic B-factors and occupancies ``(n,)``.
    A, B : torch.Tensor
        ITC92 amplitudes / widths, shape ``(n, 5)``.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Cartesian<->fractional, shape ``(3, 3)``. The truncation box comes from these, so
        there is no ``voxel_size`` argument.
    radius_per_atom : torch.Tensor
        Per-atom cutoff in Angstrom, ``(n,)``, from
        :func:`radius_policy.per_atom_radius_iso`. Used raw -- no grid-dependent
        requantization, so the cutoff means the same thing at any sampling.
    batch_offsets : torch.Tensor, optional
        ``(B + 1,)`` integer offsets assigning the flat atoms to the members of a batched
        map; required exactly when ``density_map`` is 4-D.
    """
    r2cut, offsets = _prep(
        density_map, xyz, radius_per_atom, batch_offsets, adp, occ, A, B
    )
    return _FusedIsoSplat.apply(
        density_map, xyz, adp, occ, A, B, r2cut, inv_frac_matrix, frac_matrix, offsets
    )


def add_anisotropic_cpu_sphere_var(
    density_map, xyz, u, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom,
    batch_offsets=None,
):
    """Fused anisotropic spherical-cutoff splat; returns ``density_map + splat``.

    Identical contract to :func:`add_isotropic_cpu_sphere_var`, but ``u`` carries the 6
    components ``[U11, U22, U33, U12, U13, U23]`` and the density is the full 3D Gaussian
    ``exp(-pi^2 w^T Minv w)`` with ``M_g = (B_g*I + 8*pi^2*U)/4``. The *cutoff* stays the
    Euclidean sphere at ``radius_per_atom`` (the ellipsoid's isotropic bounding radius),
    matching the CUDA and Metal kernels, which likewise cull on Euclidean distance and
    evaluate the Mahalanobis form.
    """
    r2cut, offsets = _prep(
        density_map, xyz, radius_per_atom, batch_offsets, u, occ, A, B
    )
    return _FusedAnisoSplat.apply(
        density_map, xyz, u, occ, A, B, r2cut, inv_frac_matrix, frac_matrix, offsets
    )
