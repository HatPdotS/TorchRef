"""Matrix products that stay correct on Apple M1/M2 GPUs for long reductions.

On Apple7/8 GPUs (M1, M2), the MPS matmul kernels do not clip the last tile of the
reduction dimension once it reaches 32767 and both output dimensions are at least 16:
whatever sits in memory after the operands is summed into the result. That memory is
whatever the caching allocator last held there, so the error is silent and varies
with allocator state -- a finite wrong value, or NaN. Contiguity does not help. Torch
2.14 routes plain 2-D ``mm`` around it except at K = 32767 exactly; batched ``bmm``,
3-D ``matmul`` and ``einsum`` are still affected (pytorch/pytorch#195012).

:func:`matmul` splits such a reduction into chunks short enough never to reach the
faulty tile and sums the partial products. Anywhere else it is plain ``a @ b``.

Route every product whose reduction runs over reflections, atoms or voxels through it
-- Gram and normal-equation matrices ``X.T @ (w * X)``, ``X.T @ y``, projections onto a
design -- since those reach 32767 on ordinary structures. Products that reduce over a
short axis (3x3 coordinate transforms, a few basis functions) do not need it.
"""

from __future__ import annotations

import torch

#: Reductions at least this long trigger the over-read on M1/M2.
_MPS_FAULTY_K = 32767

#: Chunk length for split reductions: well below the threshold and a power of two, so
#: every chunk except the last fills whole tiles.
_MPS_K_CHUNK = 16384


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b``, split along the reduction dimension where MPS would get it wrong.

    Parameters
    ----------
    a : torch.Tensor
        Left operand, shape ``(..., M, K)`` or ``(K,)``.
    b : torch.Tensor
        Right operand, shape ``(..., K, N)`` or ``(K,)``.

    Returns
    -------
    torch.Tensor
        ``torch.matmul(a, b)``, with ``torch.matmul``'s broadcasting and output shape.

    Notes
    -----
    On MPS with ``K >= 32767`` the product is a sum of ``ceil(K / 16384)`` partial
    products, so it differs from a single kernel call by float rounding of the
    summation order. Autograd flows through the slices as through ``a @ b``. Any
    other device, or a shorter reduction, is one ``a @ b`` call. The check is on the
    device type only: M3 and later are not affected, but chunking costs them little.
    """
    k = a.shape[-1]
    if a.device.type != "mps" or k < _MPS_FAULTY_K:
        return a @ b
    out = None
    for start in range(0, k, _MPS_K_CHUNK):
        stop = min(start + _MPS_K_CHUNK, k)
        b_part = b[start:stop] if b.dim() == 1 else b[..., start:stop, :]
        part = a[..., start:stop] @ b_part
        out = part if out is None else out + part
    return out
