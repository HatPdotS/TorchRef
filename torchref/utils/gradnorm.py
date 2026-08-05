"""Gradient-norm utilities for monitoring optimization stability."""

import torch


def gradnorm(loss: torch.Tensor, parameters: iter) -> float:
    """
    RMS of the gradients of ``loss`` with respect to ``parameters``.

    **Not read-only**: this calls ``loss.backward(retain_graph=True)``, so it *accumulates*
    into every ``.grad`` in the graph. Zero them before or after, or the next optimizer step
    uses a doubled gradient. Parameters whose grad is ``None`` are skipped, so an empty
    result raises from ``torch.cat``.

    Parameters
    ----------
    loss : torch.Tensor
        The loss tensor to backpropagate.
    parameters : iterable
        Any iterable of parameters, typically ``model.parameters()``. Consumed once, so a
        generator cannot be reused by the caller.

    Returns
    -------
    torch.Tensor
        Zero-dim tensor -- the ``-> float`` annotation is nominal, nothing is synced.
    """
    loss.backward(retain_graph=True)
    grad_norm = (
        torch.mean(
            torch.cat([p.grad.view(-1) for p in parameters if p.grad is not None]) ** 2
        )
        ** 0.5
    )
    return grad_norm
