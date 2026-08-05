"""Triton forward + analytic backward for the planarity target.

Plane normals come from a detached float64 ``eigh`` on the per-plane (P, 3, 3)
covariance, on the host: that step is not Triton-able and already runs without
autograd. Note this differs from the eager path in
``torchref.base.targets.planarity``, which uses SVD in the input dtype. The kernels
handle the per-plane gather + centroid + ``(pos - centroid)·normal`` + Gaussian NLL,
and scatter back

  ∂NLL_p/∂pos_pc = [d_pc / σ_p² - mean_a(d_pa / σ_p²)] · n_p

-- the mean term is there because ``centroid_p`` itself depends on every member atom.
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
    indices_ptr,  # (P, N_atoms) int
    normals_ptr,  # (P, 3) float
    sigmas_ptr,  # (P, N_atoms) float — per-atom sigma
    out_ptr,  # (P,) per-plane NLL
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

    # 1st pass: centroid
    cx = tl.zeros_like(nx)
    cy = tl.zeros_like(nx)
    cz = tl.zeros_like(nx)
    inv_N = 1.0 / N_ATOMS
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        cx += tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0) * inv_N
        cy += tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0) * inv_N
        cz += tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0) * inv_N

    # 2nd pass: sum of per-atom NLLs. ``sigmas`` is PER-ATOM, shape
    # (P, N_atoms) — load sigma[p, a] at ``p_offs * N_ATOMS + a``.
    nll = tl.zeros_like(nx)
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        sig_a = tl.load(sigmas_ptr + p_offs * N_ATOMS + a, mask=mask, other=1.0)
        inv_sig_a = 1.0 / sig_a
        x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
        dev = (x - cx) * nx + (y - cy) * ny + (z - cz) * nz
        nll += (
            0.5 * (dev * inv_sig_a) * (dev * inv_sig_a) + tl.log(sig_a) + 0.5 * LOG_2PI
        )
    tl.store(out_ptr + p_offs, nll, mask=mask)


@triton.jit
def _plan_nll_bwd_kernel(
    xyz_ptr,
    indices_ptr,
    normals_ptr,
    sigmas_ptr,
    grad_out_ptr,  # 0-D tensor — loaded in-kernel (no host .item() sync)
    dxyz_ptr,
    P: tl.constexpr,
    N_ATOMS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = p_offs < P
    grad_out = tl.load(grad_out_ptr)

    nx = tl.load(normals_ptr + p_offs * 3 + 0, mask=mask, other=0.0)
    ny = tl.load(normals_ptr + p_offs * 3 + 1, mask=mask, other=0.0)
    nz = tl.load(normals_ptr + p_offs * 3 + 2, mask=mask, other=0.0)
    inv_N = 1.0 / N_ATOMS

    # 1st pass: centroid
    cx = tl.zeros_like(nx)
    cy = tl.zeros_like(nx)
    cz = tl.zeros_like(nx)
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        cx += tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0) * inv_N
        cy += tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0) * inv_N
        cz += tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0) * inv_N

    # 2nd pass: mean(dev / σ²) over atoms in plane. ``sigmas`` is PER-ATOM
    # (P, N_atoms), so each term uses its own atom's sigma.
    mean_dso = tl.zeros_like(nx)
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        sig_a = tl.load(sigmas_ptr + p_offs * N_ATOMS + a, mask=mask, other=1.0)
        inv_sig2_a = 1.0 / (sig_a * sig_a)
        x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
        dev = (x - cx) * nx + (y - cy) * ny + (z - cz) * nz
        mean_dso += dev * inv_sig2_a * inv_N

    # 3rd pass: scatter gradients (per-atom sigma)
    for a in tl.static_range(N_ATOMS):
        i = tl.load(indices_ptr + p_offs * N_ATOMS + a, mask=mask, other=0)
        sig_a = tl.load(sigmas_ptr + p_offs * N_ATOMS + a, mask=mask, other=1.0)
        inv_sig2_a = 1.0 / (sig_a * sig_a)
        x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
        y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
        z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
        dev = (x - cx) * nx + (y - cy) * ny + (z - cz) * nz
        coef = grad_out * (dev * inv_sig2_a - mean_dso)
        tl.atomic_add(dxyz_ptr + i * 3 + 0, coef * nx, mask=mask)
        tl.atomic_add(dxyz_ptr + i * 3 + 1, coef * ny, mask=mask)
        tl.atomic_add(dxyz_ptr + i * 3 + 2, coef * nz, mask=mask)


def _plane_normals_via_eigh(
    covariances: torch.Tensor,
) -> torch.Tensor:
    """Smallest-eigenvalue eigenvector of a batch of 3x3 SPD covariances.

    Equivalent to the smallest right singular vector of ``centered``, since the
    covariance is ``centeredᵀ centered``, but ~5x cheaper: ``eigh`` beats ``svd`` on
    symmetric inputs and batches across plane-size buckets. Done in fp64 -- the eager
    helper's regime -- because near collinear atoms the eigenvector lives in an
    ambiguous 2-D subspace and the choice is implementation-defined.

    A NaN/Inf covariance (LBFGS does probe wild trial steps) makes ``eigh`` raise where
    SVD would return an arbitrary finite vector, so that is caught and zero normals are
    returned: deviations zero, NLL finite, gradient exactly zero, and the line search
    rejects the step on its own. Bit-perfect collinearity does not reach this path.
    """
    in_dtype = covariances.dtype
    try:
        _vals, vecs = torch.linalg.eigh(covariances.to(torch.float64))
        normals = vecs[..., 0].to(in_dtype)
    except torch._C._LinAlgError:
        # Pathological input (NaN / Inf / non-convergent). Return zero
        # normals — finite NLL, zero gradient, trial step rejected.
        normals = torch.zeros(
            covariances.shape[0],
            3,
            dtype=in_dtype,
            device=covariances.device,
        )
        return normals

    # Even when eigh succeeds it can emit NaN rows if a per-batch input
    # was finite but contained subnormals or extreme range. Same fix:
    # zero-out non-finite rows. We do the ``torch.where`` unconditionally
    # — skipping the Python ``bool(...item())`` guard so this function
    # is CUDA-Graph-capture-safe. The where is a single elementwise
    # kernel that's a no-op on the (overwhelmingly common) all-finite
    # path; cheaper than the previous host sync anyway.
    finite = torch.isfinite(normals).all(dim=-1, keepdim=True)
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
            for indices, _sigmas in plane_groups:
                positions = xyz[indices]  # (P, n, 3)
                centroids = positions.mean(dim=1, keepdim=True)
                centered = positions - centroids
                covs.append(centered.transpose(-1, -2) @ centered)  # (P, 3, 3)
                centered_per_bucket.append(centered)
            all_covs = torch.cat(covs, dim=0)  # (Σ P, 3, 3)
            all_normals = _plane_normals_via_eigh(all_covs).to(xyz.dtype)
            # Split back to per-bucket
            offset = 0
            for (_indices, _sigmas), cov in zip(plane_groups, covs):
                P = cov.shape[0]
                bucket_normals.append(all_normals[offset : offset + P].contiguous())
                offset += P

        for (indices, sigmas), normals in zip(plane_groups, bucket_normals):
            P = indices.shape[0]
            N_atoms = indices.shape[1]
            out = torch.empty(P, dtype=xyz.dtype, device=xyz.device)
            BLOCK_P = 64
            grid = (triton.cdiv(P, BLOCK_P),)
            _plan_nll_fwd_kernel[grid](
                xyz,
                indices,
                normals,
                sigmas.contiguous(),
                out,
                P=P,
                N_ATOMS=int(N_atoms),
                LOG_2PI=_LOG_2PI,
                BLOCK_P=BLOCK_P,
            )
            bucket_outs.append(out)
        ctx.save_for_backward(
            xyz, *[t for pair in plane_groups for t in pair], *bucket_normals
        )
        ctx.plane_groups = plane_groups
        ctx.bucket_normals = bucket_normals
        if not bucket_outs:
            return torch.zeros((), device=xyz.device, dtype=xyz.dtype)
        return torch.cat(bucket_outs).sum()

    @staticmethod
    def backward(ctx, grad_out):
        xyz = ctx.saved_tensors[0]
        dxyz = torch.zeros_like(xyz)
        # grad_out is a 0-D device tensor; passed by pointer.
        for (indices, sigmas), normals in zip(ctx.plane_groups, ctx.bucket_normals):
            P = indices.shape[0]
            N_atoms = indices.shape[1]
            BLOCK_P = 64
            grid = (triton.cdiv(P, BLOCK_P),)
            _plan_nll_bwd_kernel[grid](
                xyz,
                indices,
                normals,
                sigmas.contiguous(),
                grad_out,
                dxyz,
                P=P,
                N_ATOMS=int(N_atoms),
                BLOCK_P=BLOCK_P,
            )
        return dxyz, None


def planarity_math_triton(
    xyz: torch.Tensor,
    plane_groups: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Triton-backed planarity NLL with analytic backward.

    Drop-in replacement for
    :func:`torchref.base.targets.planarity.planarity_math`. Plane normals are
    computed on the host (detached) via ``eigh`` on the per-plane covariance,
    promoted to float64 then cast back; the gather + project + NLL + sum and
    the gradient scatter run in Triton.
    """
    if not plane_groups:
        return torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    return _PlanarityMathTriton.apply(xyz, plane_groups)
