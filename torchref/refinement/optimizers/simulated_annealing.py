"""
Simulated Annealing optimizer for crystallographic refinement.

Implements a per-parameter Metropolis-Hastings acceptance criterion
for fine-grained exploration of parameter space.
"""

import math
from typing import Dict, List, Literal, Union

import torch

from torchref.refinement.loss_state import LossState


def optimize_simulated_annealing(
    state: LossState,
    params: List[torch.Tensor],
    T_initial: float = 1.0,
    T_final: float = 0.01,
    n_steps: int = 1000,
    perturbation_scale: Union[float, List[float], Dict[int, float]] = 0.01,
    absolute_scale: bool = False,
    cooling_schedule: Literal["exponential", "linear"] = "exponential",
    verbose: int = 0,
    callback: callable = None,
) -> LossState:
    """
    Run simulated annealing optimization on a LossState.

    Uses per-parameter Metropolis criterion where each parameter
    is independently accepted or rejected.

    Parameters
    ----------
    state : LossState
        Configured loss state with targets and weights.
    params : list of torch.Tensor
        Parameters to optimize.
    T_initial : float
        Initial temperature. Default is 1.0.
    T_final : float
        Final temperature. Default is 0.01.
    n_steps : int
        Number of SA steps. Default is 1000.
    perturbation_scale : float, list, or dict
        Scale factor for perturbations. Can be:
        - float: uniform scale for all parameters
        - list: per-parameter scales (same length as params)
        - dict: mapping from parameter index to scale (missing indices use 0.01)
        If absolute_scale=False (default), this is multiplied by parameter magnitude.
        If absolute_scale=True, this is used directly as the standard deviation.
        Default is 0.01.
    absolute_scale : bool
        If True, perturbation_scale is used as absolute std dev for noise.
        If False (default), perturbation_scale is relative to parameter magnitude.
    cooling_schedule : str
        Cooling schedule: "exponential" or "linear". Default is "exponential".
    verbose : int
        Verbosity level. Default is 0.
    callback : callable, optional
        Function called after each step with signature callback(step, T, loss, params).
        Useful for collecting snapshots during optimization.

    Returns
    -------
    LossState
        State with history containing before/after loss values.
    """
    params = list(params)
    n_params = len(params)

    # Normalize perturbation_scale to a list
    if isinstance(perturbation_scale, (int, float)):
        scales = [float(perturbation_scale)] * n_params
    elif isinstance(perturbation_scale, list):
        if len(perturbation_scale) != n_params:
            raise ValueError(
                f"perturbation_scale list length ({len(perturbation_scale)}) "
                f"must match params length ({n_params})"
            )
        scales = [float(s) for s in perturbation_scale]
    elif isinstance(perturbation_scale, dict):
        scales = [perturbation_scale.get(i, 0.01) for i in range(n_params)]
    else:
        raise TypeError(
            f"perturbation_scale must be float, list, or dict, got {type(perturbation_scale)}"
        )

    # Log initial state
    state.aggregate(log_values=True)

    # Compute initial loss
    with torch.no_grad():
        current_loss = state.aggregate().item()

    # Cooling rate for exponential schedule
    if cooling_schedule == "exponential":
        cooling_rate = (T_final / T_initial) ** (1.0 / n_steps)

    T = T_initial
    n_accepted = 0
    n_total = 0

    for step in range(n_steps):
        # Update temperature
        if cooling_schedule == "exponential":
            T = T_initial * (cooling_rate ** step)
        else:  # linear
            T = T_initial - (T_initial - T_final) * (step / n_steps)

        # Per-parameter Metropolis steps
        for param_idx, param in enumerate(params):
            if not param.requires_grad:
                continue

            n_total += 1
            scale = scales[param_idx]

            # Save current value
            saved = param.detach().clone()

            # Perturb parameter
            with torch.no_grad():
                if absolute_scale:
                    # Use scale directly as standard deviation
                    noise_scale = scale
                else:
                    # Scale relative to parameter magnitude
                    noise_scale = scale * (param.abs().mean() + 1e-8)
                param.add_(torch.randn_like(param) * noise_scale)

            # Compute new loss
            with torch.no_grad():
                new_loss = state.aggregate().item()

            delta_E = new_loss - current_loss

            # Metropolis criterion
            if delta_E < 0:
                # Accept - lower energy
                current_loss = new_loss
                n_accepted += 1
            elif torch.rand(1).item() < math.exp(-delta_E / T):
                # Accept - probabilistic
                current_loss = new_loss
                n_accepted += 1
            else:
                # Reject - restore parameter
                with torch.no_grad():
                    param.copy_(saved)

        # Progress logging
        if verbose > 0 and (step + 1) % max(1, n_steps // 10) == 0:
            accept_rate = n_accepted / max(1, n_total)
            print(f"Step {step+1}/{n_steps}, T={T:.4f}, "
                  f"Loss={current_loss:.6f}, Accept={accept_rate:.2%}")

        # Callback for snapshots
        if callback is not None:
            callback(step, T, current_loss, params)

    # Log final state
    state.new_entry()
    state.aggregate(log_values=True)

    if verbose > 0:
        print(f"SA complete: {n_accepted}/{n_total} moves accepted "
              f"({n_accepted/max(1,n_total):.1%})")

    return state
