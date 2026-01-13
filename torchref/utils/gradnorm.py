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

    Returns
    -------
    float
        The computed RMS gradient norm.

    Notes
    -----
    Uses retain_graph=True to allow subsequent backward passes.
    Only includes parameters that have gradients (skips None grads).

    Examples
    --------
    >>> loss = model(input)
    >>> grad_norm = gradnorm(loss, model.parameters())
    >>> print(f"Gradient norm: {grad_norm:.4f}")
    """
    loss.backward(retain_graph=True)
    grad_norm = (
        torch.mean(
            torch.cat([p.grad.view(-1) for p in parameters if p.grad is not None]) ** 2
        )
        ** 0.5
    )
    return grad_norm
