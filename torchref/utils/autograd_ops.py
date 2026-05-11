"""Shared autograd helpers that wrap common ``tensor[indices]`` gathers
with cheap, deterministic backwards.

PyTorch's default backward for a 1-D ``tensor[indices]`` lowers to
``aten::_index_put_impl_(accumulate=True)``. On CUDA that path uses
``cub::DeviceRadixSortOnesweepKernel`` to sort the indices and then runs
a deduplicated scatter — a meaningful constant overhead that shows up
prominently in profiles even for small accumulator buffers (e.g.
``log_scale[bins]`` indexing into a 20-element vector).

Wrapping the gather in this autograd op routes the backward through
``index_add_`` instead: a single atomic-accumulating scatter with no
radix sort. The forward output is identical to ``buffer[indices]``.

Use via :func:`gather_with_index_add` (drop-in replacement for
``buffer[indices]`` in differentiable code paths).
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

    Drop-in replacement for the 1-D gather pattern when the forward is
    differentiable and the indices may contain duplicates.

    Parameters
    ----------
    buffer : torch.Tensor
        Source tensor (1-D, or higher-D with indexing on dim 0).
    indices : torch.Tensor
        Integer LongTensor of indices into ``buffer`` along dim 0.

    Returns
    -------
    torch.Tensor
        ``buffer[indices]`` — identical to plain indexing in forward,
        with a cheaper backward path.
    """
    return _GatherWithIndexAdd.apply(buffer, indices)
