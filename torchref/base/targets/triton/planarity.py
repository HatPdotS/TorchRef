"""Triton forward + analytic backward for the planarity target.

The plane normals are computed on the host via a detached SVD (float64,
matching the eager path) — that step is not Triton-able and already
runs without autograd. The Triton kernel handles the per-plane gather
+ centroid + (pos - centroid)·normal + Gaussian NLL + sum, and the
backward kernel scatters the analytic gradient back to ``xyz``.

Backward derivation. NLL_p = sum_a 0.5 (d_pa / σ_p)^2 + n (log σ_p + 0.5 log 2π)
where d_pa = (pos_pa - centroid_p) · n_p. Because centroid_p depends on
every atom in the plane via the mean, the gradient w.r.t. each member
atom c is

  ∂NLL_p/∂pos_pc = [d_pc / σ_p² - mean_a(d_pa / σ_p²)] · n_p .
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import triton
import triton.language as tl


_LOG_2PI = float(math.log(2.0 * math.pi))


@triton.jit
def _plan_nll_fwd_kernel(
    xyz_ptr,
    indices_ptr,     # (P, N_atoms) int
    normals_ptr,     # (P, 3) float
    sigmas_ptr,      # (P,) float
    out_ptr,         # (P,) per-plane NLL
    P: tl.constexpr,
    N_ATOMS: tl.constexpr,
    LOG_2PI: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = p_offs < P

    nx = tl.load(normals_ptr + p_offs * 3 + 0, mask=mask, other=0.0)
    ny = tl.load(normals_ptr + p_offs * 3 + 1, mask=mask, other=0.0)
    nz = tl.load(normals_ptr + p_offs * 3 + 2, mask=mask, other=0.0)
    sig = tl.load(sigmas_ptr + p_offs, mask=mask, other=1.0)
    inv_sig = 1.0 / sig

    # 1st pass: centroid
    cx = tl.zeros_like(sig)
    cy = tl.zeros_like(sig)
    cz = tl.zeros_like(sig)
    inv_N = 1.0 / N_ATOMS
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        cx += tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0) * inv_N
        cy += tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0) * inv_N
        cz += tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0) * inv_N

    # 2nd pass: sum of NLLs
    nll = tl.zeros_like(sig)
    per_atom_const = tl.log(sig) + 0.5 * LOG_2PI
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
        dev = (x - cx) * nx + (y - cy) * ny + (z - cz) * nz
        nll += 0.5 * (dev * inv_sig) * (dev * inv_sig) + per_atom_const
    tl.store(out_ptr + p_offs, nll, mask=mask)


@triton.jit
def _plan_nll_bwd_kernel(
    xyz_ptr,
    indices_ptr,
    normals_ptr,
    sigmas_ptr,
    grad_out,
    dxyz_ptr,
    P: tl.constexpr,
    N_ATOMS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = p_offs < P

    nx = tl.load(normals_ptr + p_offs * 3 + 0, mask=mask, other=0.0)
    ny = tl.load(normals_ptr + p_offs * 3 + 1, mask=mask, other=0.0)
    nz = tl.load(normals_ptr + p_offs * 3 + 2, mask=mask, other=0.0)
    sig = tl.load(sigmas_ptr + p_offs, mask=mask, other=1.0)
    inv_sig2 = 1.0 / (sig * sig)
    inv_N = 1.0 / N_ATOMS

    # 1st pass: centroid
    cx = tl.zeros_like(sig); cy = tl.zeros_like(sig); cz = tl.zeros_like(sig)
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        cx += tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0) * inv_N
        cy += tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0) * inv_N
        cz += tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0) * inv_N

    # 2nd pass: mean(dev / σ²) over atoms in plane
    mean_dso = tl.zeros_like(sig)
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
        dev = (x - cx) * nx + (y - cy) * ny + (z - cz) * nz
        mean_dso += dev * inv_sig2 * inv_N

    # 3rd pass: scatter gradients
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
        dev = (x - cx) * nx + (y - cy) * ny + (z - cz) * nz
        coef = grad_out * (dev * inv_sig2 - mean_dso)
        tl.atomic_add(dxyz_ptr + i * 3 + 0, coef * nx, mask=mask)
        tl.atomic_add(dxyz_ptr + i * 3 + 1, coef * ny, mask=mask)
        tl.atomic_add(dxyz_ptr + i * 3 + 2, coef * nz, mask=mask)


def _plane_normals_via_eigh(
    covariances: torch.Tensor,
) -> torch.Tensor:
    """Smallest-eigenvalue eigenvector of a batch of 3×3 SPD covariances.

    For an (n_atoms, 3) centered matrix, the right singular vector with
    smallest singular value coincides with the eigenvector of smallest
    eigenvalue of ``centeredᵀ centered``. Working on the (P, 3, 3)
    covariance and using ``eigh`` (faster than ``svd`` for symmetric
    inputs, batches well across plane-size buckets) cuts the
    plane-normal cost ~5× on A100 vs the per-bucket fp64 SVD on the
    full ``centered`` matrix.

    Promoted to fp64 for the eigh itself, matching the eager helper's
    precision regime — important near collinear-atom degeneracies
    where the smallest-eigvec lives in a 2-D ambiguous subspace and
    the chosen direction is implementation-defined. The fp64 cost
    is still negligible relative to the SVD it replaces.

    Robustness — LBFGS line search occasionally probes wild trial steps
    that can drive ``xyz`` (and hence the covariance) to NaN / Inf. In
    that regime ``eigh`` raises ``torch._C._LinAlgError`` whereas SVD
    on the original ``centered`` returns a finite (arbitrary) vector.
    To keep the trial step well-defined we catch the LinAlgError and
    return zero normals; the resulting deviations are zero, the NLL
    contribution is finite (``per-atom-const · n_atoms``), the gradient
    w.r.t. atom positions is exactly zero (the trial step gets a
    finite loss with no pull from planarity → validate_loss /
    line-search reject it on its own). Bit-perfect collinearity, by
    contrast, doesn't trigger this path — both ``eigh`` and ``svd``
    return a finite unit vector in the ambiguous subspace.
    """
    in_dtype = covariances.dtype
    try:
        _vals, vecs = torch.linalg.eigh(covariances.to(torch.float64))
        normals = vecs[..., 0].to(in_dtype)
    except torch._C._LinAlgError:
        # Pathological input (NaN / Inf / non-convergent). Return zero
        # normals — finite NLL, zero gradient, trial step rejected.
        normals = torch.zeros(
            covariances.shape[0], 3, dtype=in_dtype, device=covariances.device,
        )
        return normals

    # Even when eigh succeeds it can emit NaN rows if a per-batch input
    # was finite but contained subnormals or extreme range. Same fix:
    # zero-out non-finite rows.
    finite = torch.isfinite(normals).all(dim=-1, keepdim=True)
    if not bool(finite.all().item()):
        normals = torch.where(finite, normals, torch.zeros_like(normals))
    return normals


class _PlanarityMathTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, plane_groups):
        assert xyz.is_cuda and xyz.dtype == torch.float32
        # Build all per-bucket covariances, stack into one tensor, and
        # run a single batched eigh. Drops the SVD cost from 5.4 ms
        # to ~0.7 ms on 1DAW / A100.
        bucket_normals = []
        bucket_outs = []
        with torch.no_grad():
            covs = []
            centered_per_bucket = []
            for (indices, _sigmas) in plane_groups:
                positions = xyz[indices]                       # (P, n, 3)
                centroids = positions.mean(dim=1, keepdim=True)
                centered = positions - centroids
                covs.append(centered.transpose(-1, -2) @ centered)  # (P, 3, 3)
                centered_per_bucket.append(centered)
            all_covs = torch.cat(covs, dim=0)                  # (Σ P, 3, 3)
            all_normals = _plane_normals_via_eigh(all_covs).to(xyz.dtype)
            # Split back to per-bucket
            offset = 0
            for (_indices, _sigmas), cov in zip(plane_groups, covs):
                P = cov.shape[0]
                bucket_normals.append(all_normals[offset:offset + P].contiguous())
                offset += P

        for (indices, sigmas), normals in zip(plane_groups, bucket_normals):
            P = indices.shape[0]
            N_atoms = indices.shape[1]
            out = torch.empty(P, dtype=xyz.dtype, device=xyz.device)
            BLOCK_P = 64
            grid = (triton.cdiv(P, BLOCK_P),)
            _plan_nll_fwd_kernel[grid](
                xyz, indices, normals, sigmas, out,
                P=P, N_ATOMS=int(N_atoms),
                LOG_2PI=_LOG_2PI, BLOCK_P=BLOCK_P,
            )
            bucket_outs.append(out)
        ctx.save_for_backward(xyz, *[t for pair in plane_groups for t in pair],
                              *bucket_normals)
        ctx.plane_groups = plane_groups
        ctx.bucket_normals = bucket_normals
        if not bucket_outs:
            return torch.zeros((), device=xyz.device, dtype=xyz.dtype)
        return torch.cat(bucket_outs).sum()

    @staticmethod
    def backward(ctx, grad_out):
        xyz = ctx.saved_tensors[0]
        dxyz = torch.zeros_like(xyz)
        go = float(grad_out.item())
        for (indices, sigmas), normals in zip(
            ctx.plane_groups, ctx.bucket_normals
        ):
            P = indices.shape[0]
            N_atoms = indices.shape[1]
            BLOCK_P = 64
            grid = (triton.cdiv(P, BLOCK_P),)
            _plan_nll_bwd_kernel[grid](
                xyz, indices, normals, sigmas, go, dxyz,
                P=P, N_ATOMS=int(N_atoms), BLOCK_P=BLOCK_P,
            )
        return dxyz, None


def planarity_math_triton(
    xyz: torch.Tensor,
    plane_groups: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Triton-backed planarity NLL with analytic backward.

    Drop-in replacement for
    :func:`torchref.base.targets.planarity.planarity_math`. SVD-derived
    plane normals are computed on the host (detached, same as eager);
    the gather + project + NLL + sum and the gradient scatter run in
    Triton.
    """
    if not plane_groups:
        return torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    return _PlanarityMathTriton.apply(xyz, plane_groups)
