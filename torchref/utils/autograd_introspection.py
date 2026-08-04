"""
Autograd graph introspection.

Walk a loss tensor's autograd graph backward to discover the leaf
``nn.Parameter``s that gradient will accumulate into. Used by
:class:`torchref.refinement.loss_state.LossState` to record, at target
registration time, which leaves each loss touches — so the per-step
optimization path can automatically disable ``requires_grad`` on parameters
the loss depends on but the optimizer wasn't constructed with.
"""

from typing import Iterable, Mapping, Set, Union

import torch
from torch import nn

LossLike = Union[
    torch.Tensor,
    Iterable[torch.Tensor],
    Mapping[str, torch.Tensor],
]


def _iter_roots(losses: LossLike):
    """Flatten a tensor / iterable / mapping (nested, in any combination) to root tensors.

    Non-tensor entries (``None``, Python scalars) are silently skipped.
    """
    if isinstance(losses, torch.Tensor):
        yield losses
        return
    if isinstance(losses, Mapping):
        for v in losses.values():
            yield from _iter_roots(v)
        return
    if isinstance(losses, Iterable):
        for v in losses:
            yield from _iter_roots(v)
        return
    # Non-tensor, non-iterable: ignore.


def collect_loss_leaves(losses: LossLike) -> Set[nn.Parameter]:
    """The leaf ``nn.Parameter``s that ``backward()`` on ``losses`` would accumulate into.

    Walks each root's ``grad_fn`` for ``AccumulateGrad`` nodes. Multiple roots share one
    traversal, so a subgraph two losses both depend on is walked once.

    Parameters
    ----------
    losses : Tensor | Iterable[Tensor] | Mapping[str, Tensor]
        One or more loss tensors.

    Returns
    -------
    set of nn.Parameter
        A leaf with ``requires_grad=False`` is **absent** -- no ``AccumulateGrad`` node
        exists for it -- as is anything behind a ``detach()``. So an empty result means
        "nothing is currently trainable through this loss", not "the loss is constant".
    """
    # Key the seen set on the grad_fn object, never on id(): the Python wrappers from
    # ``next_functions`` are short-lived, and a reused id would skip a live node.
    seen = set()
    leaves: Set[nn.Parameter] = set()
    stack = []
    for loss in _iter_roots(losses):
        grad_fn = loss.grad_fn
        if grad_fn is not None:
            stack.append(grad_fn)
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        var = getattr(fn, "variable", None)
        if isinstance(var, nn.Parameter):
            leaves.add(var)
        for next_fn, _ in getattr(fn, "next_functions", ()):
            if next_fn is not None:
                stack.append(next_fn)
    return leaves
