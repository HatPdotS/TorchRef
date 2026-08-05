"""Shared autograd helper wrapping a 1-D ``tensor[indices]`` gather with a cheap backward.

PyTorch's default backward for that pattern lowers to
``_index_put_impl_(accumulate=True)``, which radix-sorts the indices before scattering --
a constant overhead that dominates profiles even for a 20-element accumulator like
``log_scale[bins]``. :func:`gather_with_index_add` routes the backward through
``index_add_`` instead. Correct on every device; the forward is identical to plain
indexing.

Trap: on CUDA ``index_add_`` accumulates atomically, so the gradient is **not
bit-reproducible** across runs. Anything that must be deterministic needs
``torch.use_deterministic_algorithms(True)``.
"""

from __future__ import annotations

import torch


class _GatherWithIndexAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, indices):
        ctx.save_for_backward(indices)
        ctx.buffer_shape = buffer.shape
        ctx.buffer_dtype = buffer.dtype
        ctx.buffer_device = buffer.device
        return buffer[indices]

    @staticmethod
    def backward(ctx, grad_out):
        (indices,) = ctx.saved_tensors
        grad_buffer = torch.zeros(
            ctx.buffer_shape, dtype=ctx.buffer_dtype, device=ctx.buffer_device,
        )
        # Atomic accumulating scatter along dim 0 — replaces the default
        # ``index_put_(accumulate=True)`` backward.
        grad_buffer.index_add_(0, indices, grad_out)
        return grad_buffer, None


def gather_with_index_add(
    buffer: torch.Tensor, indices: torch.Tensor,
) -> torch.Tensor:
    """``buffer[indices]`` with a fast ``index_add_`` backward.

    Drop-in for the 1-D gather pattern when the forward is differentiable and the indices
    may contain duplicates. See the module docstring for the CUDA determinism caveat.

    Parameters
    ----------
    buffer : torch.Tensor
        Source tensor (1-D, or higher-D indexed on dim 0).
    indices : torch.Tensor
        LongTensor of indices into ``buffer`` along dim 0. Not bounds-checked; the gradient
        is only defined with respect to ``buffer``.

    Returns
    -------
    torch.Tensor
        ``buffer[indices]``.
    """
    return _GatherWithIndexAdd.apply(buffer, indices)
