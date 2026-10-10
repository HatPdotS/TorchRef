"""Triton kernels for the bond-length Gaussian-NLL target.

Matches :func:`torchref.base.targets.bond.bond_math` to within float32
precision. Two kernels:

  * ``_bond_nll_fwd_kernel``: per-bond Gaussian NLL.
  * ``_bond_nll_bwd_kernel``: scatters gradients into ``xyz`` (atomic add).

They are the CUDA kernels of ``torch.ops.torchref.bond_nll_{fwd,bwd}``; the
``autograd.Function`` composing them is :data:`torchref.base.targets.ops.BondNLL`.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG_2PI = float(math.log(2.0 * math.pi))


# --------------------------------------------------------------------- kernels


@triton.jit
def _bond_nll_fwd_kernel(
    xyz_ptr,   # (N_atoms, 3) float
    idx_ptr,   # (N, 2)        int
    ref_ptr,   # (N,)          float
    sig_ptr,   # (N,)          float
    out_ptr,   # (N,)          float    -- per-bond NLL
    N: tl.constexpr,
    LOG_2PI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    i = tl.load(idx_ptr + offs * 2 + 0, mask=mask, other=0)
    j = tl.load(idx_ptr + offs * 2 + 1, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
    p2x = tl.load(xyz_ptr + j * 3 + 0, mask=mask, other=0.0)
    p2y = tl.load(xyz_ptr + j * 3 + 1, mask=mask, other=0.0)
    p2z = tl.load(xyz_ptr + j * 3 + 2, mask=mask, other=0.0)

    dx = p2x - p1x
    dy = p2y - p1y
    dz = p2z - p1z
    d = tl.sqrt(dx * dx + dy * dy + dz * dz)

    ref = tl.load(ref_ptr + offs, mask=mask, other=0.0)
    sig = tl.load(sig_ptr + offs, mask=mask, other=1.0)

    dev = d - ref
    nll = 0.5 * (dev / sig) * (dev / sig) + tl.log(sig) + 0.5 * LOG_2PI

    tl.store(out_ptr + offs, nll, mask=mask)


@triton.jit
def _bond_nll_bwd_kernel(
    xyz_ptr,         # (N_atoms, 3) -- input
    idx_ptr,         # (N, 2)
    ref_ptr,         # (N,)
    sig_ptr,         # (N,)
    grad_out_ptr,    # 0-D tensor -- gradient of upstream loss w.r.t. sum(nll).
                     # Read in-kernel so the host doesn't need to ``.item()``
                     # it (which would force a cuStreamSynchronize and block
                     # the host from queuing subsequent kernel launches).
    dxyz_ptr,        # (N_atoms, 3) -- gradient accumulator (atomic add)
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gradient of NLL_i wrt p1, p2.

    Let u_i = (p2_i - p1_i), d_i = ||u_i||, dev_i = d_i - ref_i. Then
        dNLL_i/dp2_i = (dev_i / sig_i^2) * (u_i / d_i)
        dNLL_i/dp1_i = -dNLL_i/dp2_i

    Multiplied by grad_out (= dLoss/dNLL_summed = scalar, read from
    ``grad_out_ptr`` at runtime).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # One scalar load per block, broadcast across the threads.
    grad_out = tl.load(grad_out_ptr)

    i = tl.load(idx_ptr + offs * 2 + 0, mask=mask, other=0)
    j = tl.load(idx_ptr + offs * 2 + 1, mask=mask, other=0)

    p1x = tl.load(xyz_ptr + i * 3 + 0, mask=mask, other=0.0)
    p1y = tl.load(xyz_ptr + i * 3 + 1, mask=mask, other=0.0)
    p1z = tl.load(xyz_ptr + i * 3 + 2, mask=mask, other=0.0)
    p2x = tl.load(xyz_ptr + j * 3 + 0, mask=mask, other=0.0)
    p2y = tl.load(xyz_ptr + j * 3 + 1, mask=mask, other=0.0)
    p2z = tl.load(xyz_ptr + j * 3 + 2, mask=mask, other=0.0)

    ux = p2x - p1x
    uy = p2y - p1y
    uz = p2z - p1z
    d = tl.sqrt(ux * ux + uy * uy + uz * uz)
    d_safe = tl.where(d > 0, d, 1.0)  # avoid 0/0 at degenerate pairs

    ref = tl.load(ref_ptr + offs, mask=mask, other=0.0)
    sig = tl.load(sig_ptr + offs, mask=mask, other=1.0)

    dev = d - ref
    coef = grad_out * dev / (sig * sig) / d_safe  # scalar per bond

    gx = coef * ux
    gy = coef * uy
    gz = coef * uz

    # Scatter-add into the gradient buffer. Two atoms per bond, opposite signs.
    tl.atomic_add(dxyz_ptr + j * 3 + 0, gx, mask=mask)
    tl.atomic_add(dxyz_ptr + j * 3 + 1, gy, mask=mask)
    tl.atomic_add(dxyz_ptr + j * 3 + 2, gz, mask=mask)
    tl.atomic_add(dxyz_ptr + i * 3 + 0, -gx, mask=mask)
    tl.atomic_add(dxyz_ptr + i * 3 + 1, -gy, mask=mask)
    tl.atomic_add(dxyz_ptr + i * 3 + 2, -gz, mask=mask)


# --------------------------------------------------------------- autograd wrap


def bond_nll_fwd(xyz: torch.Tensor, idx: torch.Tensor,
                 references: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """CUDA kernel of ``torch.ops.torchref.bond_nll_fwd``: the summed bond NLL.

    ``xyz`` must be row-major ``(N_atoms, 3)`` with stride ``(3, 1)``; ``MixedTensor``
    guarantees it (``parameter_wrappers.py``).
    """
    assert xyz.is_cuda and idx.is_cuda
    assert xyz.dtype == torch.float32
    N = idx.shape[0]
    nll = torch.empty(N, dtype=xyz.dtype, device=xyz.device)
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    _bond_nll_fwd_kernel[grid](
        xyz, idx, references, sigmas, nll,
        N=N, LOG_2PI=_LOG_2PI, BLOCK=BLOCK,
    )
    return nll.sum()


def bond_nll_bwd(grad_out: torch.Tensor, xyz: torch.Tensor, idx: torch.Tensor,
                 references: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """CUDA kernel of ``torch.ops.torchref.bond_nll_bwd``: ``grad_out * d NLL / d xyz``."""
    N = idx.shape[0]
    dxyz = torch.zeros_like(xyz)
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    # grad_out goes in as a 0-D device tensor (its data_ptr): the kernel ``tl.load``s it
    # once per block -- no ``.item()``, no host sync.
    _bond_nll_bwd_kernel[grid](
        xyz, idx, references, sigmas, grad_out, dxyz,
        N=N, BLOCK=BLOCK,
    )
    return dxyz


def bond_math_triton(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    references: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Triton-backed bond-length Gaussian NLL.

    Drop-in replacement for :func:`torchref.base.targets.bond.bond_math`.
    """
    from ..ops import BondNLL

    return BondNLL.apply(xyz, idx, references, sigmas)
