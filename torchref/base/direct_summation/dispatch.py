"""Backend dispatch for direct-summation structure factors.

Selection uses the shared, capability-based :class:`~torchref.utils.Engine`
(see :mod:`torchref.utils.triton_dispatch`): for a given ``(device, dtype)``
there is exactly one best path, so the backend is *derived* rather than
configured.

- CUDA + float32 + Triton available  ->  custom Triton kernels (``triton_ds``)
- everything else (CPU, float64, MPS, no Triton)  ->  checkpointed eager

An explicit ``Engine`` override (per-call ``engine=`` or the process-wide
``use_engine``/``set_engine``) forces a path for tests and benchmarks.

All backends compute a **P1** structure-factor sum only. Crystallographic
symmetry is applied outside, in :meth:`SfDS.compute_structure_factors`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from torchref.base.direct_summation.isotropic import (
    _estimate_batch_size,
    iso_structure_factor_torched,
)
from torchref.base.direct_summation.anisotropic import (
    _estimate_batch_size_aniso,
    aniso_structure_factor_torched,
)
from torchref.utils.triton_dispatch import Engine, should_use_triton

TWO_PI = 2.0 * math.pi
NEG_TWO_PI_SQ = -2.0 * (math.pi**2)


# ---------------------------------------------------------------------------
# Lazy Triton probe (mirrors electron_density/main.py:_get_*_triton)
# ---------------------------------------------------------------------------
_triton_iso_fn = None
_triton_aniso_fn = None
_triton_checked = False


def _load_triton():
    """Import the Triton kernels once; leave the fns ``None`` if unavailable."""
    global _triton_iso_fn, _triton_aniso_fn, _triton_checked
    if not _triton_checked:
        try:
            from torchref.base.direct_summation.triton_ds import (
                ds_iso_triton,
                ds_aniso_triton,
            )

            _triton_iso_fn = ds_iso_triton
            _triton_aniso_fn = ds_aniso_triton
        except Exception:
            pass
        _triton_checked = True


# ---------------------------------------------------------------------------
# P1 identity symmetry (lives here, not in sf_ds)
# ---------------------------------------------------------------------------
def _p1_symmetry(coords_3N: torch.Tensor) -> torch.Tensor:
    """Identity 'symmetry': (3, N) -> (3, N, 1)."""
    return coords_3N.unsqueeze(2)


# ---------------------------------------------------------------------------
# Plain-torch chunk math — single source of truth for the checkpointed path
# ---------------------------------------------------------------------------
def _scattering_factors(s_mag: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """ITC92 5-Gaussian f(s) for a reflection chunk: (R_c, N)."""
    s_sq = (s_mag.reshape(-1, 1, 1) ** 2) / 4  # (R_c, 1, 1)
    exp_terms = torch.exp(-B.unsqueeze(0) * s_sq)  # (R_c, N, 5)
    return torch.sum(A.unsqueeze(0) * exp_terms, dim=-1)  # (R_c, N)


def _chunk_math_iso(hkl_c, s_c, xyz_frac, occ, adp, A, B):
    """Real/imag P1 structure factor for a reflection chunk (isotropic).

    Forms only a ``(R_c, N)`` tile. Returns ``(Fr_c, Fi_c)`` of shape ``(R_c,)``.
    """
    f = _scattering_factors(s_c, A, B)  # (R_c, N)
    dw = torch.exp(-adp.reshape(1, -1) * (s_c.reshape(-1, 1) ** 2) / 4)  # (R_c, N)
    c = occ.reshape(1, -1) * f * dw  # (R_c, N)
    phi = TWO_PI * torch.matmul(hkl_c.to(xyz_frac.dtype), xyz_frac.T)  # (R_c, N)
    Fr = torch.sum(c * torch.cos(phi), dim=1)
    Fi = torch.sum(c * torch.sin(phi), dim=1)
    return Fr, Fi


def _chunk_math_aniso(hkl_c, s_vec_c, xyz_frac, occ, U, A, B):
    """Real/imag P1 structure factor for a reflection chunk (anisotropic)."""
    s_mag = torch.norm(s_vec_c, dim=1)  # (R_c,)
    f = _scattering_factors(s_mag, A, B)  # (R_c, N)
    sx, sy, sz = s_vec_c[:, 0], s_vec_c[:, 1], s_vec_c[:, 2]  # (R_c,)
    # sT U s with U=[U11,U22,U33,U12,U13,U23]; broadcast (R_c,1)*(1,N)
    sUs = (
        (sx**2).reshape(-1, 1) * U[:, 0].reshape(1, -1)
        + (sy**2).reshape(-1, 1) * U[:, 1].reshape(1, -1)
        + (sz**2).reshape(-1, 1) * U[:, 2].reshape(1, -1)
        + 2 * (sx * sy).reshape(-1, 1) * U[:, 3].reshape(1, -1)
        + 2 * (sx * sz).reshape(-1, 1) * U[:, 4].reshape(1, -1)
        + 2 * (sy * sz).reshape(-1, 1) * U[:, 5].reshape(1, -1)
    )  # (R_c, N)
    dw = torch.exp(NEG_TWO_PI_SQ * sUs)  # (R_c, N)
    c = occ.reshape(1, -1) * f * dw  # (R_c, N)
    phi = TWO_PI * torch.matmul(hkl_c.to(xyz_frac.dtype), xyz_frac.T)  # (R_c, N)
    Fr = torch.sum(c * torch.cos(phi), dim=1)
    Fi = torch.sum(c * torch.sin(phi), dim=1)
    return Fr, Fi


def _chunk_ranges(n_refl: int, n_atoms: int, max_memory_gb: Optional[float], bytes_per: int):
    """Yield (start, end) reflection chunks bounded by ``max_memory_gb``."""
    if max_memory_gb is None:
        yield 0, n_refl
        return
    max_bytes = max_memory_gb * 1e9
    chunk = max(1, int(max_bytes / max(1, n_atoms * bytes_per)))
    chunk = min(chunk, n_refl)
    for start in range(0, n_refl, chunk):
        yield start, min(start + chunk, n_refl)


# ---------------------------------------------------------------------------
# Checkpointed eager backend — recompute on backward, no large intermediate
# ---------------------------------------------------------------------------
class _CheckpointedSF(torch.autograd.Function):
    """Chunked, recompute-on-backward structure factor (iso or aniso).

    Forward runs under ``no_grad`` accumulating directly into ``Fr/Fi`` and
    saves only the small inputs (never an ``R x N`` tile). Backward replays
    each chunk under ``enable_grad`` and uses ``torch.autograd.grad`` to obtain
    per-chunk parameter gradients — so the differentiation is exact (autograd)
    while peak memory stays at one chunk.
    """

    @staticmethod
    def forward(ctx, chunk_fn, geom, max_memory_gb, bytes_per, xyz_frac, occ, adp_or_U, A, B):
        # ``geom`` is hkl_c-source (hkl) plus s/s_vec; passed as a tuple so it
        # is not differentiated. Refinable leaves are explicit args.
        hkl, s_or_svec = geom
        n_refl = hkl.shape[0]
        n_atoms = xyz_frac.shape[0]
        device = hkl.device
        Fr = torch.zeros(n_refl, dtype=xyz_frac.dtype, device=device)
        Fi = torch.zeros(n_refl, dtype=xyz_frac.dtype, device=device)
        with torch.no_grad():
            for start, end in _chunk_ranges(n_refl, n_atoms, max_memory_gb, bytes_per):
                Fr_c, Fi_c = chunk_fn(
                    hkl[start:end], s_or_svec[start:end], xyz_frac, occ, adp_or_U, A, B
                )
                Fr[start:end] = Fr_c
                Fi[start:end] = Fi_c
        ctx.chunk_fn = chunk_fn
        ctx.max_memory_gb = max_memory_gb
        ctx.bytes_per = bytes_per
        ctx.save_for_backward(hkl, s_or_svec, xyz_frac, occ, adp_or_U, A, B)
        return torch.complex(Fr, Fi)

    @staticmethod
    def backward(ctx, grad_F):
        hkl, s_or_svec, xyz_frac, occ, adp_or_U, A, B = ctx.saved_tensors
        gFr = grad_F.real.contiguous()
        gFi = grad_F.imag.contiguous()
        n_refl = hkl.shape[0]
        n_atoms = xyz_frac.shape[0]

        leaves = [xyz_frac, occ, adp_or_U]
        needs = [t.requires_grad for t in leaves]
        accum = [
            torch.zeros_like(t) if need else None for t, need in zip(leaves, needs)
        ]
        wanted = [t for t, need in zip(leaves, needs) if need]

        if wanted:
            for start, end in _chunk_ranges(
                n_refl, n_atoms, ctx.max_memory_gb, ctx.bytes_per
            ):
                with torch.enable_grad():
                    xc = xyz_frac.detach().requires_grad_(needs[0])
                    oc = occ.detach().requires_grad_(needs[1])
                    pc = adp_or_U.detach().requires_grad_(needs[2])
                    Fr_c, Fi_c = ctx.chunk_fn(
                        hkl[start:end], s_or_svec[start:end], xc, oc, pc, A, B
                    )
                    chunk_leaves = [
                        v
                        for v, need in zip((xc, oc, pc), needs)
                        if need
                    ]
                    grads = torch.autograd.grad(
                        (Fr_c, Fi_c),
                        chunk_leaves,
                        grad_outputs=(gFr[start:end], gFi[start:end]),
                        allow_unused=True,
                    )
                gi = 0
                for k, need in enumerate(needs):
                    if not need:
                        continue
                    g = grads[gi]
                    gi += 1
                    if g is not None:
                        accum[k] = accum[k] + g

        # arg order: chunk_fn, geom, max_memory_gb, bytes_per, xyz, occ, adp_or_U, A, B
        return (None, None, None, None, accum[0], accum[1], accum[2], None, None)


def _checkpointed_iso(hkl, s, xyz_frac, occ, adp, A, B, max_memory_gb):
    return _CheckpointedSF.apply(
        _chunk_math_iso, (hkl, s), max_memory_gb, 50, xyz_frac, occ, adp, A, B
    )


def _checkpointed_aniso(hkl, s_vec, xyz_frac, occ, U, A, B, max_memory_gb):
    return _CheckpointedSF.apply(
        _chunk_math_aniso, (hkl, s_vec), max_memory_gb, 80, xyz_frac, occ, U, A, B
    )


# ---------------------------------------------------------------------------
# Eager (autograd) backend — the existing reference implementation
# ---------------------------------------------------------------------------
def _eager_iso(hkl, s, xyz_frac, occ, adp, A, B, max_memory_gb):
    return iso_structure_factor_torched(
        hkl=hkl, s=s, xyz_fractional=xyz_frac, occ=occ,
        scattering_factors=None, adp=adp, spacegroup=_p1_symmetry,
        max_memory_gb=max_memory_gb, A=A, B_coeff=B,
    )


def _eager_aniso(hkl, s_vec, xyz_frac, occ, U, A, B, max_memory_gb):
    return aniso_structure_factor_torched(
        hkl=hkl, s_vector=s_vec, xyz_fractional=xyz_frac, occ=occ,
        scattering_factors=None, U=U, spacegroup=_p1_symmetry,
        max_memory_gb=max_memory_gb, A=A, B_coeff=B,
    )


# ---------------------------------------------------------------------------
# Dispatch entry points
# ---------------------------------------------------------------------------
# The coarse Triton-vs-eager gate is the shared ``should_use_triton`` (device,
# dtype, engine). The DS-kernel-module availability is then checked here: under
# AUTO a missing/failing kernel falls back to checkpointed eager; under
# Engine.TRITON it raises.
def ds_iso(hkl, s, xyz_frac, occ, adp, A, B, *, engine=Engine.AUTO, max_memory_gb=None):
    """Isotropic P1 structure factors via the selected backend."""
    if xyz_frac.shape[0] == 0:
        return torch.zeros(hkl.shape[0], dtype=torch.complex64, device=hkl.device)
    if should_use_triton(xyz_frac, engine=engine):
        _load_triton()
        if _triton_iso_fn is not None:
            try:
                return _triton_iso_fn(hkl, s, xyz_frac, occ, adp, A, B)
            except Exception:
                if engine is Engine.TRITON:
                    raise
                # AUTO: fall back to checkpointed eager
        elif engine is Engine.TRITON:
            raise RuntimeError("Direct-summation Triton kernels are unavailable")
    return _checkpointed_iso(hkl, s, xyz_frac, occ, adp, A, B, max_memory_gb)


def ds_aniso(hkl, s_vec, xyz_frac, occ, U, A, B, *, engine=Engine.AUTO, max_memory_gb=None):
    """Anisotropic P1 structure factors via the selected backend."""
    if xyz_frac.shape[0] == 0:
        return torch.zeros(hkl.shape[0], dtype=torch.complex64, device=hkl.device)
    if should_use_triton(xyz_frac, engine=engine):
        _load_triton()
        if _triton_aniso_fn is not None:
            try:
                return _triton_aniso_fn(hkl, s_vec, xyz_frac, occ, U, A, B)
            except Exception:
                if engine is Engine.TRITON:
                    raise
        elif engine is Engine.TRITON:
            raise RuntimeError("Direct-summation Triton kernels are unavailable")
    return _checkpointed_aniso(hkl, s_vec, xyz_frac, occ, U, A, B, max_memory_gb)
