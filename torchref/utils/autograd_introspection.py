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
    """Flatten losses to an iterable of root tensors.

    Accepts a single ``Tensor``, a tuple/list of tensors, a dict whose
    values are tensors, or any nested combination thereof. Non-tensor
    entries (``None``, Python scalars, etc.) are silently skipped.
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
    """Return the set of leaf ``nn.Parameter``s that gradient will
    accumulate into when ``backward()`` is called on the given loss(es).

    Walks the autograd graph from each root tensor's ``grad_fn`` and
    finds every ``AccumulateGrad`` node, collecting its ``.variable``
    when it is an :class:`nn.Parameter`.

    Multiple roots are unioned via a single shared traversal so that
    shared subgraphs (e.g. two losses both depending on the same model
    forward) are walked exactly once.

    Parameters
    ----------
    losses : Tensor | Iterable[Tensor] | Mapping[str, Tensor]
        One or more loss tensors.

    Returns
    -------
    set of nn.Parameter
        Leaf parameters that backward would accumulate gradient into.
        A leaf with ``requires_grad=False`` does not appear (no
        ``AccumulateGrad`` node is created for it). Detached subtrees
        contribute nothing.
    """
    # Use the grad_fn object itself as the seen key. grad_fn instances are
    # hashable and equality-compares to the same underlying C++ Node, so this
    # is correct. We deliberately avoid id()-keying because Python wrapper
    # objects returned by ``next_functions`` are short-lived — once popped
    # off the stack and out of scope, their id can be reused for an unrelated
    # wrapper, causing the seen set to incorrectly skip live nodes.
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
