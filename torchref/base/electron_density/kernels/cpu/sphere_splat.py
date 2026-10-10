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
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.base.electron_density.ops import SplatAniso, SplatIso
from torchref.utils import native


class _RustSplat:
    """Tensor-level adapter over the ``torchref-kernels`` splat entry points.

    Every tensor must already be contiguous and of ``out``'s / ``go``'s dtype;
    :func:`torchref.utils.native.buf` checks that before any pointer is handed over.
    """

    def __init__(self, mod):
        self._mod = mod

    def _fn(self, name, dtype):
        suffix = {torch.float32: "f32", torch.float64: "f64"}.get(dtype)  # dtype-ok: kernel dtype dispatch, not an allocation
        if suffix is None:
            raise ValueError(f"sphere_splat supports float32/float64, got {dtype}")
        return getattr(self._mod, f"sphere_{name}_{suffix}")

    def _call(self, name, outs, ins, nx, ny, nz):
        dtype = outs[0][1].dtype
        bufs = [native.buf(t, dtype, n) for n, t in outs + ins]
        self._fn(name, dtype)(*bufs, nx, ny, nz, native.num_threads())

    def iso_fwd(self, out, xyz, adp, occ, A, B, r2cut, inv_frac, frac, nx, ny, nz):
        self._call(
            "iso_fwd", [("out", out)],
            [("xyz", xyz), ("adp", adp), ("occ", occ), ("A", A), ("B", B),
             ("r2cut", r2cut), ("inv_frac", inv_frac), ("frac", frac)],
            nx, ny, nz,
        )

    def iso_bwd(self, g_xyz, g_adp, g_occ, go, xyz, adp, occ, A, B, r2cut,
                inv_frac, frac, nx, ny, nz):
        self._call(
            "iso_bwd", [("g_xyz", g_xyz), ("g_adp", g_adp), ("g_occ", g_occ)],
            [("grad", go), ("xyz", xyz), ("adp", adp), ("occ", occ), ("A", A),
             ("B", B), ("r2cut", r2cut), ("inv_frac", inv_frac), ("frac", frac)],
            nx, ny, nz,
        )

    def aniso_fwd(self, out, xyz, u, occ, A, B, r2cut, inv_frac, frac, nx, ny, nz):
        self._call(
            "aniso_fwd", [("out", out)],
            [("xyz", xyz), ("u", u), ("occ", occ), ("A", A), ("B", B),
             ("r2cut", r2cut), ("inv_frac", inv_frac), ("frac", frac)],
            nx, ny, nz,
        )

    def aniso_bwd(self, g_xyz, g_u, g_occ, go, xyz, u, occ, A, B, r2cut,
                  inv_frac, frac, nx, ny, nz):
        self._call(
            "aniso_bwd", [("g_xyz", g_xyz), ("g_u", g_u), ("g_occ", g_occ)],
            [("grad", go), ("xyz", xyz), ("u", u), ("occ", occ), ("A", A),
             ("B", B), ("r2cut", r2cut), ("inv_frac", inv_frac), ("frac", frac)],
            nx, ny, nz,
        )


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


def _double_backward_vjp(plain_fn, ctx, grad_out, leaves, statics, r2cut):
    """Recompute this VJP through the portable splat for a ``create_graph=True`` backward.

    The native backward has no autograd graph, so :class:`~torchref.base.electron_density.ops.SplatIso`
    and ``SplatAniso`` call this on CPU instead when a second derivative may be taken. Gradients are taken with respect to the saved (not
    detached) leaves, so the result stays on the caller's graph; the gradient with respect
    to ``density_map`` is the identity.
    """
    A, B, inv_frac, frac = statics
    # Only the leaves that actually require grad may be differentiated; asking for
    # the others raises regardless of allow_unused.
    wanted = [t for t in leaves if t.requires_grad]
    out = [None] * len(leaves)
    if wanted:
        zeros = torch.zeros(
            ctx.grid_shape, dtype=grad_out.dtype, device=grad_out.device
        )
        with torch.enable_grad():
            dm = plain_fn(zeros, *leaves, A, B, inv_frac, frac, r2cut.sqrt())
        grads = torch.autograd.grad(
            dm, wanted, grad_out, create_graph=True, allow_unused=True
        )
        it = iter(grads)
        out = [next(it) if t.requires_grad else None for t in leaves]
    # forward returned density_map + splat -> grad wrt density_map is the identity
    return (grad_out, *out) + (None,) * 5


def _prep(density_map, xyz, radius_per_atom, *tensors):
    """Shared validation + contiguity for both entry points."""
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
    return dtype, r2cut


def _flat_mats(inv_frac, frac):
    return inv_frac.contiguous().view(-1), frac.contiguous().view(-1)


def iso_fwd(density_map, xyz, adp, occ, A, B, r2cut, inv_frac, frac):
    """CPU kernel of ``torch.ops.torchref.splat_iso_fwd``: returns ``density_map + splat``."""
    nx, ny, nz = (int(s) for s in density_map.shape)
    out = density_map.contiguous().clone()
    if xyz.shape[0] > 0:
        im, fm = _flat_mats(inv_frac, frac)
        _require_module().iso_fwd(
            out.view(-1), xyz.contiguous(), adp.contiguous(), occ.contiguous(),
            A.contiguous(), B.contiguous(), r2cut.contiguous(), im, fm, nx, ny, nz,
        )
    return out


def iso_bwd(grad, xyz, adp, occ, A, B, r2cut, inv_frac, frac):
    """CPU kernel of ``torch.ops.torchref.splat_iso_bwd``: ``(d xyz, d adp, d occ)``."""
    nx, ny, nz = (int(s) for s in grad.shape)
    g_xyz, g_adp, g_occ = (torch.zeros_like(t) for t in (xyz, adp, occ))
    if xyz.shape[0] > 0:
        im, fm = _flat_mats(inv_frac, frac)
        _require_module().iso_bwd(
            g_xyz.view(-1), g_adp, g_occ, grad.contiguous().view(-1),
            xyz.contiguous(), adp.contiguous(), occ.contiguous(), A.contiguous(),
            B.contiguous(), r2cut.contiguous(), im, fm, nx, ny, nz,
        )
    return g_xyz, g_adp, g_occ


def aniso_fwd(density_map, xyz, u, occ, A, B, r2cut, inv_frac, frac):
    """CPU kernel of ``torch.ops.torchref.splat_aniso_fwd``: returns ``density_map + splat``."""
    nx, ny, nz = (int(s) for s in density_map.shape)
    out = density_map.contiguous().clone()
    if xyz.shape[0] > 0:
        im, fm = _flat_mats(inv_frac, frac)
        _require_module().aniso_fwd(
            out.view(-1), xyz.contiguous(), u.contiguous(), occ.contiguous(),
            A.contiguous(), B.contiguous(), r2cut.contiguous(), im, fm, nx, ny, nz,
        )
    return out


def aniso_bwd(grad, xyz, u, occ, A, B, r2cut, inv_frac, frac):
    """CPU kernel of ``torch.ops.torchref.splat_aniso_bwd``: ``(d xyz, d u, d occ)``."""
    nx, ny, nz = (int(s) for s in grad.shape)
    g_xyz, g_u, g_occ = (torch.zeros_like(t) for t in (xyz, u, occ))
    if xyz.shape[0] > 0:
        im, fm = _flat_mats(inv_frac, frac)
        _require_module().aniso_bwd(
            g_xyz.view(-1), g_u.view(-1), g_occ, grad.contiguous().view(-1),
            xyz.contiguous(), u.contiguous(), occ.contiguous(), A.contiguous(),
            B.contiguous(), r2cut.contiguous(), im, fm, nx, ny, nz,
        )
    return g_xyz, g_u, g_occ


def add_isotropic_cpu_sphere_var(
    density_map, xyz, adp, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Fused isotropic spherical-cutoff splat; returns ``density_map + splat``.

    Parameters
    ----------
    density_map : torch.Tensor
        Running map, shape ``(nx, ny, nz)``, CPU float32/float64. **Not mutated.**
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
    """
    _, r2cut = _prep(density_map, xyz, radius_per_atom, adp, occ, A, B)
    return SplatIso.apply(
        density_map, xyz, adp, occ, A, B, r2cut, inv_frac_matrix, frac_matrix
    )


def add_anisotropic_cpu_sphere_var(
    density_map, xyz, u, occ, A, B, inv_frac_matrix, frac_matrix, radius_per_atom
):
    """Fused anisotropic spherical-cutoff splat; returns ``density_map + splat``.

    Identical contract to :func:`add_isotropic_cpu_sphere_var`, but ``u`` carries the 6
    components ``[U11, U22, U33, U12, U13, U23]`` and the density is the full 3D Gaussian
    ``exp(-pi^2 w^T Minv w)`` with ``M_g = (B_g*I + 8*pi^2*U)/4``. The *cutoff* stays the
    Euclidean sphere at ``radius_per_atom`` (the ellipsoid's isotropic bounding radius),
    matching the CUDA and Metal kernels, which likewise cull on Euclidean distance and
    evaluate the Mahalanobis form.
    """
    _, r2cut = _prep(density_map, xyz, radius_per_atom, u, occ, A, B)
    return SplatAniso.apply(
        density_map, xyz, u, occ, A, B, r2cut, inv_frac_matrix, frac_matrix
    )
