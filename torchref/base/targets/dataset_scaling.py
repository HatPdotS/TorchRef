"""Profile a shared amplitude against independently measured datasets.

The inputs use a common reflection axis with explicit presence masks. Uncertainties
remain in their measurement units; scaling and consensus estimation are differentiable.
"""

import torch


def dataset_scaling_loss(
    amplitudes: torch.Tensor,
    sigmas: torch.Tensor,
    log_corrections: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the summed profiled Gaussian least-squares loss.

    Parameters
    ----------
    amplitudes, sigmas : torch.Tensor
        Measured amplitudes and positive uncertainties, shape (N, H), in each
        dataset's input amplitude units. Masked entries may be non-finite.
    log_corrections : torch.Tensor
        Dimensionless log amplitude corrections, shape (N, H).
    mask : torch.Tensor
        Boolean usable-observation mask, shape (N, H). Only columns with at
        least two observations contribute.

    Returns
    -------
    torch.Tensor
        Scalar dimensionless loss. Gradients include the consensus and the
        scale dependence of the propagated uncertainties.
    """
    active = mask & (mask.sum(dim=0, keepdim=True) >= 2)
    obs = torch.where(active, amplitudes, torch.zeros_like(amplitudes))
    sigma = torch.where(active, sigmas, torch.ones_like(sigmas))
    log_k = torch.where(active, log_corrections, torch.zeros_like(log_corrections))
    # In measurement units the model is mu / k and sigma is fixed. Rescaling
    # its whitened design column avoids overflow without changing the projection.
    log_design = -log_k - sigma.log()
    floor = torch.finfo(log_design.dtype).min
    shift = torch.where(active, log_design, floor).amax(dim=0, keepdim=True)
    shift = torch.where(active.any(dim=0, keepdim=True), shift, torch.zeros_like(shift))
    shifted = torch.where(active, log_design - shift, torch.zeros_like(obs))
    design = torch.where(active, shifted.exp(), torch.zeros_like(obs))
    whitened = obs / sigma
    norm = design.square().sum(dim=0).clamp_min(torch.finfo(design.dtype).tiny)
    consensus = (design * whitened).sum(dim=0) / norm
    residual = torch.where(active, whitened - design * consensus, torch.zeros_like(obs))
    return 0.5 * residual.square().sum()
