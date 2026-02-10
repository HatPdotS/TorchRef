"""
Stochastic Simulated Annealing optimizer for internal coordinates.

Implements per-parameter Metropolis-Hastings with stochastic subset selection,
designed for SegmentedInternalCoordinateTensor where many parameters are
approximately independent due to localized rigid body segments.
"""

import math
from typing import Dict, Optional, Callable

import torch
import torch.nn as nn

from torchref.refinement.loss_state import LossState


def optimize_stochastic_sa(
    state: LossState,
    internal_coords: nn.Module,
    T_initial: float = 100.0,
    T_final: float = 0.01,
    n_steps: int = 5000,
    fraction_per_step: float = 0.05,
    perturbation_scales: Optional[Dict[str, float]] = None,
    cooling_schedule: str = "exponential",
    verbose: int = 0,
    callback: Optional[Callable] = None,
) -> LossState:
    """
    Stochastic Simulated Annealing for internal coordinates.

    Each step:
    1. Randomly select a fraction of parameters from each type
    2. Perturb each selected parameter independently
    3. Evaluate loss once (exploiting independence)
    4. Accept/reject each perturbation based on contribution

    Since internal coordinates in different segments are largely independent,
    we can perturb multiple parameters and evaluate their combined effect,
    then use per-parameter acceptance decisions.

    Parameters
    ----------
    state : LossState
        Configured loss state with targets.
    internal_coords : nn.Module
        SegmentedInternalCoordinateTensor or InternalCoordinateTensor.
        Must have attributes: bond_lengths, angles, torsions,
        segment_positions, segment_orientations.
    T_initial : float
        Initial temperature. Should be calibrated to loss scale.
        For losses ~1000, try T_initial=100-1000.
    T_final : float
        Final temperature.
    n_steps : int
        Number of SA steps.
    fraction_per_step : float
        Fraction of each parameter type to perturb per step.
        E.g., 0.05 = 5% of torsions perturbed each step.
    perturbation_scales : dict, optional
        Per-parameter-type scales. Defaults:
        - 'bond': 0.02 Å
        - 'angle': 0.02 rad (~1°)
        - 'torsion': 0.1 rad (~6°)
        - 'position': 0.1 Å
        - 'orientation': 0.05 rad (~3°)
    cooling_schedule : str
        'exponential' or 'linear'.
    verbose : int
        Verbosity level.
    callback : callable, optional
        Called after each step: callback(step, T, loss, accept_rates).

    Returns
    -------
    LossState
        Updated state.
    """
    # Default perturbation scales (physically motivated)
    # Bonds and angles should have very small perturbations (they're stiff)
    # Torsions are the main flexible degrees of freedom
    default_scales = {
        'bond': 0.005,        # Å - bonds are very stiff
        'angle': 0.005,       # rad (~0.3°) - angles are stiff
        'torsion': 0.03,      # rad (~2°) - torsions are flexible
        'position': 0.02,     # Å - small segment position shifts
        'orientation': 0.01,  # rad (~0.6°) - small orientation changes
    }
    if perturbation_scales is not None:
        default_scales.update(perturbation_scales)
    scales = default_scales

    # Get parameter references
    params_info = []

    if hasattr(internal_coords, 'bond_lengths') and internal_coords.bond_lengths.numel() > 0:
        params_info.append({
            'name': 'bond',
            'param': internal_coords.bond_lengths,
            'scale': scales['bond'],
            'min_val': 0.5,  # Minimum bond length
            'max_val': 5.0,  # Maximum bond length
        })

    if hasattr(internal_coords, 'angles') and internal_coords.angles.numel() > 0:
        params_info.append({
            'name': 'angle',
            'param': internal_coords.angles,
            'scale': scales['angle'],
            'min_val': 0.1,           # ~6°
            'max_val': math.pi - 0.1, # ~174°
        })

    if hasattr(internal_coords, 'torsions') and internal_coords.torsions.numel() > 0:
        params_info.append({
            'name': 'torsion',
            'param': internal_coords.torsions,
            'scale': scales['torsion'],
            'wrap': True,  # Torsions wrap around
        })

    if hasattr(internal_coords, 'segment_positions') and internal_coords.segment_positions.numel() > 0:
        params_info.append({
            'name': 'position',
            'param': internal_coords.segment_positions,
            'scale': scales['position'],
        })

    if hasattr(internal_coords, 'segment_orientations') and internal_coords.segment_orientations.numel() > 0:
        params_info.append({
            'name': 'orientation',
            'param': internal_coords.segment_orientations,
            'scale': scales['orientation'],
        })

    if len(params_info) == 0:
        raise ValueError("No refinable internal coordinate parameters found")

    # Print parameter summary
    if verbose > 0:
        print("Stochastic SA Parameter Summary:")
        total_params = 0
        for p in params_info:
            n = p['param'].numel()
            total_params += n
            print(f"  {p['name']:12s}: {n:6d} params, scale={p['scale']:.4f}")
        print(f"  {'TOTAL':12s}: {total_params:6d} params")
        print(f"  Fraction per step: {fraction_per_step:.1%} ({int(total_params * fraction_per_step)} params)")

    # Cooling rate
    if cooling_schedule == "exponential":
        cooling_rate = (T_final / T_initial) ** (1.0 / n_steps)

    # Initialize
    with torch.no_grad():
        current_loss = state.aggregate().item()

    T = T_initial

    # Tracking
    n_accepted = {p['name']: 0 for p in params_info}
    n_proposed = {p['name']: 0 for p in params_info}

    for step in range(n_steps):
        # Update temperature
        if cooling_schedule == "exponential":
            T = T_initial * (cooling_rate ** step)
        else:
            T = T_initial - (T_initial - T_final) * (step / n_steps)

        # For each parameter type
        for pinfo in params_info:
            param = pinfo['param']
            scale = pinfo['scale']
            name = pinfo['name']

            # Determine number of parameters to perturb
            n_total = param.numel()
            n_perturb = max(1, int(n_total * fraction_per_step))

            # Randomly select indices to perturb
            flat_param = param.view(-1)
            indices = torch.randperm(n_total, device=param.device)[:n_perturb]

            # Process each selected parameter
            for idx in indices:
                n_proposed[name] += 1

                # Save current value
                with torch.no_grad():
                    saved_val = flat_param[idx].clone()

                    # Perturb
                    noise = torch.randn(1, device=param.device, dtype=param.dtype).item() * scale
                    flat_param[idx] += noise

                    # Apply constraints
                    if pinfo.get('wrap', False):
                        # Wrap torsions to [-π, π]
                        flat_param[idx] = torch.atan2(
                            torch.sin(flat_param[idx]),
                            torch.cos(flat_param[idx])
                        )
                    elif 'min_val' in pinfo:
                        flat_param[idx] = torch.clamp(
                            flat_param[idx],
                            min=pinfo['min_val'],
                            max=pinfo.get('max_val', float('inf'))
                        )

                # Evaluate new loss
                with torch.no_grad():
                    new_loss = state.aggregate().item()

                delta_E = new_loss - current_loss

                # Metropolis criterion
                accept = False
                if delta_E < 0:
                    accept = True
                elif T > 0 and torch.rand(1).item() < math.exp(-delta_E / T):
                    accept = True

                if accept:
                    current_loss = new_loss
                    n_accepted[name] += 1
                else:
                    # Reject - restore
                    with torch.no_grad():
                        flat_param[idx] = saved_val

        # Progress logging
        if verbose > 0 and (step + 1) % max(1, n_steps // 10) == 0:
            total_accepted = sum(n_accepted.values())
            total_proposed = sum(n_proposed.values())
            accept_rate = total_accepted / max(1, total_proposed)
            print(f"Step {step+1}/{n_steps}, T={T:.4f}, Loss={current_loss:.4f}, "
                  f"Accept={accept_rate:.1%}")

            if verbose > 1:
                for name in n_accepted:
                    if n_proposed[name] > 0:
                        rate = n_accepted[name] / n_proposed[name]
                        print(f"    {name}: {rate:.1%}")

        # Callback
        if callback is not None:
            accept_rates = {k: n_accepted[k]/max(1, n_proposed[k]) for k in n_accepted}
            callback(step, T, current_loss, accept_rates)

    # Final summary
    if verbose > 0:
        print("\nStochastic SA Complete:")
        print(f"  Final loss: {current_loss:.4f}")
        print("  Acceptance rates by parameter type:")
        for name in n_accepted:
            if n_proposed[name] > 0:
                rate = n_accepted[name] / n_proposed[name]
                print(f"    {name}: {n_accepted[name]}/{n_proposed[name]} ({rate:.1%})")

    return state


def optimize_stochastic_sa_batch(
    state: LossState,
    internal_coords: nn.Module,
    T_initial: float = 100.0,
    T_final: float = 0.01,
    n_steps: int = 5000,
    fraction_per_step: float = 0.05,
    perturbation_scales: Optional[Dict[str, float]] = None,
    cooling_schedule: str = "exponential",
    verbose: int = 0,
    callback: Optional[Callable] = None,
) -> LossState:
    """
    Batch variant: perturb multiple parameters, evaluate once, accept/reject together.

    This is faster but less fine-grained than the per-parameter version.
    Good approximation when parameters are largely independent (segmented internal coords).

    Parameters are same as optimize_stochastic_sa.
    """
    # Default perturbation scales (same as per-parameter version)
    default_scales = {
        'bond': 0.005,        # Å - bonds are very stiff
        'angle': 0.005,       # rad (~0.3°) - angles are stiff
        'torsion': 0.03,      # rad (~2°) - torsions are flexible
        'position': 0.02,     # Å - small segment position shifts
        'orientation': 0.01,  # rad (~0.6°) - small orientation changes
    }
    if perturbation_scales is not None:
        default_scales.update(perturbation_scales)
    scales = default_scales

    # Get parameter references
    params_info = []

    if hasattr(internal_coords, 'bond_lengths') and internal_coords.bond_lengths.numel() > 0:
        params_info.append({
            'name': 'bond',
            'param': internal_coords.bond_lengths,
            'scale': scales['bond'],
            'min_val': 0.5,
            'max_val': 5.0,
        })

    if hasattr(internal_coords, 'angles') and internal_coords.angles.numel() > 0:
        params_info.append({
            'name': 'angle',
            'param': internal_coords.angles,
            'scale': scales['angle'],
            'min_val': 0.1,
            'max_val': math.pi - 0.1,
        })

    if hasattr(internal_coords, 'torsions') and internal_coords.torsions.numel() > 0:
        params_info.append({
            'name': 'torsion',
            'param': internal_coords.torsions,
            'scale': scales['torsion'],
            'wrap': True,
        })

    if hasattr(internal_coords, 'segment_positions') and internal_coords.segment_positions.numel() > 0:
        params_info.append({
            'name': 'position',
            'param': internal_coords.segment_positions,
            'scale': scales['position'],
        })

    if hasattr(internal_coords, 'segment_orientations') and internal_coords.segment_orientations.numel() > 0:
        params_info.append({
            'name': 'orientation',
            'param': internal_coords.segment_orientations,
            'scale': scales['orientation'],
        })

    if len(params_info) == 0:
        raise ValueError("No refinable internal coordinate parameters found")

    # Print summary
    if verbose > 0:
        print("Stochastic SA (Batch) Parameter Summary:")
        total_params = 0
        for p in params_info:
            n = p['param'].numel()
            total_params += n
            print(f"  {p['name']:12s}: {n:6d} params, scale={p['scale']:.4f}")
        print(f"  Fraction per step: {fraction_per_step:.1%}")

    # Cooling
    if cooling_schedule == "exponential":
        cooling_rate = (T_final / T_initial) ** (1.0 / n_steps)

    # Initialize
    with torch.no_grad():
        current_loss = state.aggregate().item()

    T = T_initial
    n_accepted = 0
    n_proposed = 0

    for step in range(n_steps):
        # Update temperature
        if cooling_schedule == "exponential":
            T = T_initial * (cooling_rate ** step)
        else:
            T = T_initial - (T_initial - T_final) * (step / n_steps)

        # Save all current values
        saved_values = {}
        perturbation_masks = {}

        with torch.no_grad():
            for pinfo in params_info:
                param = pinfo['param']
                name = pinfo['name']

                # Save
                saved_values[name] = param.data.clone()

                # Create perturbation mask
                n_total = param.numel()
                n_perturb = max(1, int(n_total * fraction_per_step))
                mask = torch.zeros(n_total, dtype=torch.bool, device=param.device)
                indices = torch.randperm(n_total, device=param.device)[:n_perturb]
                mask[indices] = True
                perturbation_masks[name] = mask.view(param.shape)

                # Apply perturbation
                noise = torch.randn_like(param) * pinfo['scale']
                param.data += noise * perturbation_masks[name]

                # Constraints
                if pinfo.get('wrap', False):
                    param.data = torch.atan2(torch.sin(param.data), torch.cos(param.data))
                elif 'min_val' in pinfo:
                    param.data = torch.clamp(param.data, min=pinfo['min_val'],
                                             max=pinfo.get('max_val', float('inf')))

        # Evaluate
        n_proposed += 1
        with torch.no_grad():
            new_loss = state.aggregate().item()

        delta_E = new_loss - current_loss

        # Metropolis
        accept = False
        if delta_E < 0:
            accept = True
        elif T > 0 and torch.rand(1).item() < math.exp(-delta_E / T):
            accept = True

        if accept:
            current_loss = new_loss
            n_accepted += 1
        else:
            # Restore all
            with torch.no_grad():
                for pinfo in params_info:
                    pinfo['param'].data.copy_(saved_values[pinfo['name']])

        # Progress
        if verbose > 0 and (step + 1) % max(1, n_steps // 10) == 0:
            accept_rate = n_accepted / max(1, n_proposed)
            print(f"Step {step+1}/{n_steps}, T={T:.4f}, Loss={current_loss:.4f}, "
                  f"Accept={accept_rate:.1%}")

        if callback is not None:
            callback(step, T, current_loss, {'overall': n_accepted/max(1, n_proposed)})

    if verbose > 0:
        print(f"\nBatch SA Complete: {n_accepted}/{n_proposed} steps accepted "
              f"({n_accepted/max(1, n_proposed):.1%})")

    return state
