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
kernels. The backward is differentiable once more: under ``create_graph=True`` it is
recorded as ``_SphereSplatGrad``, whose backward runs two native second-order passes, the
tangent map (partitioned by x-plane, like the forward) and each atom's Hessian-vector
product (over atoms, like the backward). A third derivative raises.

A batched call (``(B, nx, ny, nz)`` map plus ``batch_offsets``, see
:mod:`~torchref.base.electron_density.batching`) runs each member through the same
kernels, so it is bitwise the stack of the members' unbatched results.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from torchref.base.electron_density.batching import check_batch, dense_offsets
from torchref.base.targets._dispatch import first_order_only
from torchref.utils import native


_INPUTS = ("xyz", "adp", "occ", "A", "B", "r2cut", "inv_frac", "frac")
_TANGENTS = ("t_xyz", "t_adp", "t_occ")
_ARG_NAMES = {
    "fwd": ("out",) + _INPUTS,
    "bwd": ("g_xyz", "g_adp", "g_occ", "grad") + _INPUTS,
    "tan": ("out",) + _TANGENTS + _INPUTS,
    "hvp": ("h_xyz", "h_adp", "h_occ", "grad") + _TANGENTS + _INPUTS,
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
    iso_tan = functools.partialmethod(_call, "iso_tan")
    iso_hvp = functools.partialmethod(_call, "iso_hvp")
    aniso_fwd = functools.partialmethod(_call, "aniso_fwd")
    aniso_bwd = functools.partialmethod(_call, "aniso_bwd")
    aniso_tan = functools.partialmethod(_call, "aniso_tan")
    aniso_hvp = functools.partialmethod(_call, "aniso_hvp")


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
        offsets = dense_offsets(1, xyz.shape[0])
    return r2cut, offsets.contiguous()


def _zeros_contiguous(t):
    # Contiguous whatever the input strides: the kernels write these as flat arrays.
    return torch.zeros_like(t, memory_format=torch.contiguous_format)


def _statics(xyz, third, occ, A, B, r2cut, inv_frac, frac):
    """The kernels' read-only inputs, as the contiguous buffers they take."""
    return (
        xyz.detach().contiguous(), third.detach().contiguous(),
        occ.detach().contiguous(), A.contiguous(), B.contiguous(),
        r2cut, inv_frac.contiguous().view(-1), frac.contiguous().view(-1),
    )


class _SphereSplatGrad(torch.autograd.Function):
    """The splat's atom gradients as a function of the map gradient and the atoms.

    The backward of both fused splats, ``kind`` naming the entry points (``"iso"`` or
    ``"aniso"``). Autograd records it under ``create_graph=True``, and its own backward
    runs the native tangent and Hessian-vector passes; that backward is first-order only.
    """

    @staticmethod
    def forward(ctx, grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac, offsets,
                kind, grid_shape):
        grads = tuple(_zeros_contiguous(t) for t in (xyz, third, occ))
        if xyz.shape[0] > 0:
            getattr(_require_module(), f"{kind}_bwd")(
                grads[0].view(-1), grads[1].view(-1), grads[2],
                grad_out.contiguous().view(-1),
                *_statics(xyz, third, occ, A, B, r2cut, inv_frac, frac),
                offsets, *grid_shape,
            )
        ctx.kind, ctx.grid_shape = kind, grid_shape
        ctx.save_for_backward(grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac,
                              offsets)
        return grads

    @staticmethod
    @first_order_only
    def backward(ctx, t_xyz, t_third, t_occ):
        grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac, offsets = ctx.saved_tensors
        atoms = (xyz, third, occ)
        # A cotangent autograd leaves out (None) is zero.
        cotangents = tuple(
            _zeros_contiguous(x) if t is None else t.detach().contiguous()
            for t, x in zip((t_xyz, t_third, t_occ), atoms)
        )
        cotangents = (cotangents[0].view(-1), cotangents[1].view(-1), cotangents[2])
        hvps = tuple(_zeros_contiguous(x) for x in atoms)
        # The tangent map is the gradient wrt grad_out, wanted only when the map gradient
        # itself depends on the atoms (any loss nonlinear in the density).
        tangent = _zeros_contiguous(grad_out) if ctx.needs_input_grad[0] else None
        if xyz.shape[0] > 0:
            mod = _require_module()
            statics = _statics(xyz, third, occ, A, B, r2cut, inv_frac, frac)
            getattr(mod, f"{ctx.kind}_hvp")(
                hvps[0].view(-1), hvps[1].view(-1), hvps[2],
                grad_out.contiguous().view(-1), *cotangents, *statics,
                offsets, *ctx.grid_shape,
            )
            if tangent is not None:
                getattr(mod, f"{ctx.kind}_tan")(
                    tangent.view(-1), *cotangents, *statics, offsets, *ctx.grid_shape,
                )
        # order: grad_out, xyz, third, occ, A, B, r2cut, inv_frac, frac, offsets, kind,
        # grid_shape
        return (tangent, *hvps) + (None,) * 8


def _forward(ctx, kind, density_map, xyz, third, occ, A, B, r2cut, inv_frac, frac,
             offsets):
    """Forward shared by both splats; returns ``density_map + splat``."""
    mod = _require_module()
    nx, ny, nz = (int(s) for s in density_map.shape[-3:])
    out = density_map.contiguous().clone()
    if xyz.shape[0] > 0:
        getattr(mod, f"{kind}_fwd")(
            out.view(-1), *_statics(xyz, third, occ, A, B, r2cut, inv_frac, frac),
            offsets, nx, ny, nz,
        )
    ctx.save_for_backward(xyz, third, occ, A, B, r2cut, inv_frac, frac, offsets)
    ctx.grid_shape = (nx, ny, nz)
    return out


def _backward(ctx, kind, grad_out):
    """Backward shared by both splats; the gradient of ``third`` has its shape."""
    grads = _SphereSplatGrad.apply(grad_out, *ctx.saved_tensors, kind, ctx.grid_shape)
    # out = density_map + splat, so grad wrt density_map is the identity.
    return (grad_out, *grads, None, None, None, None, None, None)


class _FusedIsoSplat(torch.autograd.Function):
    """Isotropic fused CPU splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, adp, occ, A, B, r2cut, inv_frac, frac,
                offsets):
        return _forward(ctx, "iso", density_map, xyz, adp, occ, A, B, r2cut, inv_frac,
                        frac, offsets)

    @staticmethod
    def backward(ctx, grad_out):
        return _backward(ctx, "iso", grad_out)


class _FusedAnisoSplat(torch.autograd.Function):
    """Anisotropic fused CPU splat: returns ``density_map + splat``."""

    @staticmethod
    def forward(ctx, density_map, xyz, u, occ, A, B, r2cut, inv_frac, frac,
                offsets):
        return _forward(ctx, "aniso", density_map, xyz, u, occ, A, B, r2cut, inv_frac,
                        frac, offsets)

    @staticmethod
    def backward(ctx, grad_out):
        return _backward(ctx, "aniso", grad_out)


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
