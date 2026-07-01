"""
Gradient norm utilities for optimization monitoring.

This module provides functions to compute gradient norms for monitoring
training stability and debugging optimization issues.
"""

import torch


def gradnorm(loss: torch.Tensor, parameters: iter) -> float:
    """
    Compute the gradient norm of a loss with respect to given parameters.

    Performs a backward pass with graph retention and computes the RMS
    (root mean square) of all gradients concatenated together.

    Parameters
    ----------
    loss : torch.Tensor
        The loss tensor to backpropagate.
    parameters : iterable
        Iterable of model parameters (typically from model.parameters()).
        Note the runtime annotation reads ``iter``; the expected argument
        is any iterable of parameters.

    Returns
    -------
    torch.Tensor
        The computed RMS gradient norm as a zero-dim ``torch.Tensor``.
        (The ``-> float`` annotation is nominal; the value is returned as a
        scalar tensor, not a Python ``float``.)

    Notes
    -----
    Has a side effect: this function calls ``loss.backward(retain_graph=True)``,
    which populates / accumulates the ``.grad`` attribute on the parameters.
    ``retain_graph=True`` allows subsequent backward passes.
    Only includes parameters that have gradients (skips None grads).

    Examples
    --------
    ::

        loss = model(input)
        grad_norm = gradnorm(loss, model.parameters())
        print(f"Gradient norm: {grad_norm:.4f}")
    """
    loss.backward(retain_graph=True)
    grad_norm = (
        torch.mean(
            torch.cat([p.grad.view(-1) for p in parameters if p.grad is not None]) ** 2
        )
        ** 0.5
    )
    return grad_norm
